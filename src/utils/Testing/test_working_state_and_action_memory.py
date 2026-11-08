from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from controllers.history_controller import HistoryController
from managers.action_memory import (
    append_requested_actions,
    cap_requested_actions,
    requested_actions_from_structured,
    render_requested_actions,
)
from managers.working_state_manager import WorkingStateManager
from schemas.structured_response import StructuredResponse, WorkingState


class _Character:
    def __init__(self) -> None:
        self.variables: dict[str, object] = {}
        self.char_id = "Test"

    def get_variable(self, key, default=None):
        return self.variables.get(key, default)


class WorkingStateAndActionMemoryTests(unittest.TestCase):
    def test_working_state_is_bounded_and_rendered_separately(self):
        manager = WorkingStateManager().bind("Mita", "Mita")
        manager.update(
            WorkingState(
                focus="Follow the dance test",
                situation=["Dance_07 was requested", "The player is testing continuity"],
                assumptions=["The player may want a practical answer"],
                open_loops=["Check whether the music matters"],
                next_steps=["Respond from the active test"],
            ),
            max_chars=400,
        )
        prompt = manager.format_for_prompt()
        self.assertIn("[WORKING STATE]", prompt)
        self.assertIn("Focus: Follow the dance test", prompt)
        self.assertNotIn("chain-of-thought", prompt)
        manager.clear()
        self.assertEqual(manager.format_for_prompt(), "")

    def test_missing_working_state_preserves_previous_scene(self):
        manager = WorkingStateManager().bind("Crazy", "Crazy")
        manager.update(WorkingState(
            focus="Keep playing the cat game",
            situation=["Player and Mita are testing an ability"],
        ))
        response = StructuredResponse.model_validate({
            "segments": [{"text": "The game continues."}],
        })
        self.assertIsNone(response.working_state)
        self.assertNotIn("working_state", response.model_fields_set)
        self.assertEqual(manager.commit_from_response(response), "retained")
        self.assertIn("Keep playing the cat game", manager.format_for_prompt())

    def test_explicit_null_clears_previous_scene(self):
        manager = WorkingStateManager().bind("Kind", "Kind")
        manager.update(WorkingState(focus="Wait for the next move"))
        response = StructuredResponse.model_validate({
            "segments": [{"text": "That scene is finished."}],
            "working_state": None,
        })
        self.assertIn("working_state", response.model_fields_set)
        self.assertEqual(manager.commit_from_response(response), "cleared")
        self.assertEqual(manager.format_for_prompt(), "")

    def test_new_scene_replaces_previous_and_respects_char_limit(self):
        manager = WorkingStateManager().bind("ShortHair", "ShortHair")
        manager.update(WorkingState(focus="Old activity"))
        response = StructuredResponse.model_validate({
            "segments": [{"text": "The next game starts."}],
            "working_state": {
                "focus": "New scene with another Mita",
                "situation": ["We are continuing a shared activity with the player." * 20],
                "assumptions": ["A claim about a cognitive effect needs runtime proof."],
            },
        })
        self.assertEqual(manager.commit_from_response(response, max_chars=200), "updated")
        output = manager.format_for_prompt()
        self.assertIn("New scene with another Mita", output)
        self.assertNotIn("Old activity", output)
        self.assertLessEqual(len(output.split("[/WORKING STATE]")[0]), 400)

    def test_empty_working_state_object_does_not_discard_scene(self):
        manager = WorkingStateManager().bind("Cappie", "Cappie")
        manager.update(WorkingState(focus="Stay with the scene"))
        response = StructuredResponse.model_validate({
            "segments": [{"text": "I'm here."}],
            "working_state": {},
        })
        self.assertEqual(manager.commit_from_response(response), "empty")
        self.assertIn("Stay with the scene", manager.format_for_prompt())

    def test_action_projection_uses_requested_structured_actions_only(self):
        records = requested_actions_from_structured({
            "segments": [{
                "text": "Смотри!",
                "animations": ["Dance_07"],
                "commands": ["music:track_3"],
                "intents": [{"type": "inventory.collect", "payload": {"id": "key"}}],
            }],
            "memory_add": ["normal|not an action"],
        })
        self.assertEqual(records, [
            "animation: Dance_07",
            "command: music:track_3",
            'intent: inventory.collect {"id":"key"}',
        ])
        rendered = render_requested_actions(records)
        self.assertIn("[RECENT ACTIONS BEFORE SUMMARY BOUNDARY]", rendered)
        self.assertIn("chronological order", rendered)
        self.assertNotIn("performed", rendered.lower())

    def test_action_projection_normalizes_legacy_clothing_commands(self):
        records = requested_actions_from_structured({
            "segments": [{
                "commands": [
                    "clothes:VampClothes,vamp_nebula_turquoise",
                    "ClothesColor,All,250,250,250",
                    "ClothesColor,Skirt,0,255,127",
                    "ClothesColor,reset",
                ],
            }],
        })
        self.assertEqual(records, [
            "outfit: VampClothes, vamp_nebula_turquoise",
            "appearance recolor: All rgb(250,250,250)",
            "appearance recolor: Skirt rgb(0,255,127)",
            "appearance recolor: reset",
        ])

    def test_action_emergency_cap_preserves_newest_suffix(self):
        records, capped = cap_requested_actions(
            ["animation: Dance_01", "animation: Dance_02", "animation: Dance_03"],
            max_records=2,
            max_chars=200,
        )
        self.assertTrue(capped)
        self.assertEqual(records, ["animation: Dance_02", "animation: Dance_03"])

        records, capped = cap_requested_actions(
            ["x" * 300],
            max_records=10,
            max_chars=200,
        )
        self.assertTrue(capped)
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].endswith("… [truncated]"))

    def test_action_tail_changes_only_when_summary_commits(self):
        controller = HistoryController.__new__(HistoryController)
        controller._get_setting = lambda key, default=None: {
            "ENABLE_ACTION_MEMORY": True,
            "ACTION_MEMORY_RETAIN_LAST": 2,
        }.get(key, default)
        character = _Character()
        character.variables[controller._ACTION_MEMORY_RETAINED_VAR] = '["animation: Old"]'
        compressed = [
            {"role": "assistant", "structured_data": {"segments": [{"animations": ["Dance_01"]}]}},
            {"role": "assistant", "structured_data": {"segments": [{"animations": ["Dance_02"]}]}},
        ]
        retained = controller._next_retained_action_requests(character, compressed)
        self.assertEqual(retained, ["animation: Dance_01", "animation: Dance_02"])
        character.variables[controller._ACTION_MEMORY_RETAINED_VAR] = retained
        context = controller._build_action_memory_context(character)
        self.assertIn("Dance_01", context)
        self.assertIn("Dance_02", context)
        self.assertNotIn("Dance_03", context)

    def test_recent_actions_stay_on_their_assistant_reply(self):
        controller = HistoryController.__new__(HistoryController)
        controller._get_setting = lambda key, default=None: {
            "ENABLE_ACTION_MEMORY": True,
            "HISTORY_TIME_GAP_MARKERS": False,
        }.get(key, default)
        original = {
            "role": "assistant",
            "content": "Давай попробуем этот.",
            "structured_data": {"segments": [{"animations": ["Dance_07"]}]},
        }
        projected = controller._sanitize_history_for_llm(_Character(), [original])

        self.assertEqual(original["content"], "Давай попробуем этот.")
        self.assertEqual(projected[0]["role"], "assistant")
        self.assertIn("Давай попробуем этот.", projected[0]["content"])
        self.assertIn("[REQUESTED ACTIONS THIS TURN]", projected[0]["content"])
        self.assertIn("animation: Dance_07", projected[0]["content"])

        controller._get_setting = lambda _key, default=None: default
        disabled = controller._sanitize_history_for_llm(_Character(), [original])
        self.assertNotIn("[REQUESTED ACTIONS THIS TURN]", disabled[0]["content"])

    def test_action_projection_appends_after_multimodal_message(self):
        projected = append_requested_actions(
            [{"type": "text", "text": "Смотри"}, {"type": "image_url", "image_url": {"url": "x"}}],
            ["animation: Dance_07"],
        )
        self.assertEqual(projected[-1]["type"], "text")
        self.assertIn("[REQUESTED ACTIONS THIS TURN]", projected[-1]["text"])

    def test_working_state_can_be_removed_from_provider_schema(self):
        full = StructuredResponse.openai_response_format()["json_schema"]["schema"]
        without = StructuredResponse.openai_response_format(
            exclude_fields={"working_state"},
        )["json_schema"]["schema"]
        self.assertIn("working_state", full["properties"])
        self.assertNotIn("working_state", without["properties"])


if __name__ == "__main__":
    unittest.main()
