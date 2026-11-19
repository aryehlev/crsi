"""Seed harness: hand the task to the model and return whatever it says.

It does nothing about answer format and gives the model no way to check its work. That is the
starting point the optimizer improves.
"""


def run(task, llm):
    return llm.complete("You are a helpful assistant.", task["prompt"])
