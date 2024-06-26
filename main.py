import spacy
from g2p_en import G2p
import copy
from openai import OpenAI
import urllib
from string import Template
import json
import jsons
import random
import sys
from pathlib import Path
import re
import argparse

from storyer import Character, Scene, Story, StorySource

class CharacterProcessor:
    def __init__(self):
        self.characters = {}

class SceneProcessor:
    def __init__(self):
        self.scenes = []

class StoryProcessor:
    def __init__(self):
        self.stories = []


def create_story(name: str) -> Story:
    story = Story(name=name + " " + str(random.randint(0, 1000)))
    story.scenes = [create_scene() for _ in range(3)]
    return story

def create_scene() -> Scene:
    return Scene( text="scene " + str(random.randint(0, 1000)) )

def read_json_file_db() -> list[Story]:
    with open("db.json", "r") as file:
        return jsons.loads(file.read())

def save_json_file_db(stories: list[Story]) -> None:
    with open("db.json", "w") as file:
        file.write(jsons.dumps(stories, jdkwargs={'indent': 4}))


def main():
    parser = argparse.ArgumentParser(description="Visualize a story.")
    parser.add_argument('story_name', type=str, help='The name of the story in stories/_story_/ directory')
    parser.add_argument('--force', type=str, default='true', help='force proessessing of the story even if it already exists in the db')
    args = parser.parse_args()

    name = args.story_name
    force = args.force

    source = StorySource(story_name)
    story = source.load_story()
    if ( force == 'true' ):
        # re-proces the story
        pass
    else:
        # only process if not previously procesed
        pass

    print("\n".join(str(story) for story in stories))
    save_json_file_db(stories)

if __name__ == "__main__":
    main()
