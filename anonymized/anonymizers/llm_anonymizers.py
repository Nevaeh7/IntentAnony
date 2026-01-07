from abc import ABC, abstractmethod
from typing import List, Iterator, Dict, Any
from copy import deepcopy
from .anonymizer import Anonymizer

import sys
from pathlib import Path
from typing import Any

# Add project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.string_utils import select_closest
from configs import Config, AnonymizerConfig
from prompts import Prompt
from reddit.reddit_utils import type_to_str
from reddit.reddit_types import Comment, AnnotatedComments
import re


class LLMFullAnonymizer(Anonymizer):
    def __init__(
        self,
        cfg: AnonymizerConfig,
        model: Any,
    ):
        self.model = model
        self.cfg = cfg

    def _create_anon_prompt(self, profile: Any) -> List[Any]:

        if self.cfg.prompt_level == 1:  # Naive
            system_prompt = "You are an helpful assistant that helps users to anonymize their comments."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments."
            footer = "In a new line write first a single # and then return the anonymized text."
        elif self.cfg.prompt_level == 2:  # Better prompt and explanation
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a new line write first a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information."
        elif self.cfg.prompt_level == 3:  # CoT
            system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
            header = "Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact."
            footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_inferences = profile.get_latest_comments().predictions[
            self.model.config.name
        ]
        inference_string = ""
        try:
            for key, inf in previous_inferences.items():
                if key == "full_answer":
                    continue
                if "guess" not in inf:
                    continue
                inference_string += f"Type: {key}\n"
                inference_string += f"Inference: {inf['inference']}\n"
                inference_string += f"Guess: {inf['guess']}\n"
        except Exception as e:
            # Fall back to full answer
            inference_string = previous_inferences["full_answer"]

        intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=(
                profile.get_relevant_pii()[0]
                if len(profile.get_relevant_pii()) > 0
                else ""
            ),
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]

    def filter_and_align_comments(self, answer: str, op: Dict[str, Any]) -> List[str]:

        try:
            split_answer = answer.split("\n#")

            if len(split_answer) == 1:
                new_comments = answer.strip()
            elif len(split_answer) == 2:
                new_comments = split_answer[1].strip()
            else:
                new_comments = ("\n").join(split_answer)

        except Exception:
            print("Could not split answer", answer)
            new_comments = deepcopy([c.text for c in op.get_latest_comments().comments])
            return new_comments

        new_comments = new_comments.split("\n")

        # Remove all lines that are empty
        new_comments = [c for c in new_comments if len(c) > 0]

        if len(new_comments) != len(op.get_latest_comments().comments):
            print(
                f"Number of comments does not match: {len(new_comments)} vs {len(op.get_latest_comments().comments)}"
            )

            old_comment_ids = [
                -1 for _ in range(len(op.get_latest_comments().comments))
            ]

            used_idx = set({})

            for i, comment in enumerate(op.get_latest_comments().comments):
                closest_match, sim, idx = select_closest(
                    comment.text,
                    new_comments,
                    dist="jaro_winkler",
                    return_idx=True,
                    return_sim=True,
                )

                if idx not in used_idx and sim > 0.5:
                    old_comment_ids[i] = idx
                    used_idx.add(idx)

            selected_comments = []
            for i, idx in enumerate(old_comment_ids):
                if idx == -1:
                    selected_comments.append(op.get_latest_comments().comments[i].text)
                else:
                    selected_comments.append(new_comments[idx])
        else:
            selected_comments = new_comments

        typed_comments = []
        i = 0

        for comment in selected_comments:
            if re.search(r"\d{4}-\d{2}-\d{2}:", comment[:11]) is not None:
                comment = comment[11:].strip()

            old_com = op.get_latest_comments().comments[i]
            new_com = Comment(
                comment, old_com.subreddit, old_com.user, old_com.timestamp
            )
            typed_comments.append(new_com)
            i += 1

        return typed_comments

    def anonymize(self, text: str) -> str:
        pass

    def anonymize_profiles(self, profiles: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:

        prompts = []
        for profile in profiles:
            prompts.extend(self._create_anon_prompt(profile))

        for i, res in enumerate(
            self.model.predict_multi(
                prompts, max_workers=self.cfg.max_workers, timeout=120
            )
        ):
            prompt, answer = res
            # for i, prompt in enumerate(prompts):
            #     answer = self.model.predict(prompt)

            op = prompt.original_point
            print(f"{i}".center(50, "="))
            # print(prompt.get_prompt())
            op.print_review_pii()
            # print(f"{self.model.config.name}\n" + answer)

            typed_comments = self.filter_and_align_comments(answer, op)

            # print(typed_comments)

            op.comments.append(AnnotatedComments(typed_comments, op.review_pii, {}, {}))

            yield op


class LLMBaselineAnonymizer(LLMFullAnonymizer):

    def _create_anon_prompt(self, profile: Dict[str, Any]) -> List[Any]:

        type_str = ""
        for pii in profile.get_relevant_pii():
            type_str += f"{type_to_str(pii)} "

        system_prompt = "You are an expert anonymizer that carefully adapts small parts of the text to make it anonymous. You follow the instructions and format precisely and you try to change as little as possible, keeping the original text in tact as much as possible."
        header = f"Below I give you a set of comments from a user. Your task is to anonymize the comments to prevent the inferences of the users {type_str} , while keeping the rest of the text intact."
        footer = "In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized text. Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information."

        comments = profile.get_latest_comments().comments
        comment_string = "\n".join([str(c) for c in comments])
        previous_infereces = profile.get_latest_comments().predictions[
            self.model.config.name
        ]

        intermediate = f"\n\n {comment_string}\n"

        prompt = Prompt(
            system_prompt=system_prompt,
            header=header,
            intermediate=intermediate,
            footer=footer,
            target=profile.get_relevant_pii()[0],
            original_point=profile,  # type: ignore
            gt=profile.get_relevant_pii(),  # type: ignore
            answer="",
            shots=[],
            id=profile.username,  # type: ignore
        )

        return [prompt]
