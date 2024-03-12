from rioplatense_hs.prompting import FewShotPromptTemplate


def test_generate_template_with_no_examples():
    instruction = "This is an instruction."

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["input"],
        output_variables=["output"],
        examples=[],
        separator="###",
    )
    assert (
        prompt.get(input="This is an input", examples=[])
        == "This is an instruction.\ninput: This is an input"
    )


def test_generate_template_with_some_examples():
    instruction = "This is an instruction."
    examples = [
        {
            "input": "This is an input.",
            "output": "This is an output.",
        },
        {
            "input": "This is another input.",
            "output": "This is another output.",
        },
    ]

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["input"],
        output_variables=["output"],
        separator="###",
        examples=examples,
    )

    prompt = prompt.get(input="This is an input", examples=examples)

    expected_prompt = """This is an instruction.
###
input: This is an input.
output: This is an output.
###
input: This is another input.
output: This is another output.
###
input: This is an input"""
    assert prompt == expected_prompt


def test_generate_template_with_two_input_variables():
    instruction = "This is an instruction."
    examples = [
        {
            "input1": "This is an input.",
            "input2": "This is another input.",
            "output": "This is an output.",
        },
    ]

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["input1", "input2"],
        output_variables=["output"],
        separator="###",
        examples=examples,
    )

    prompt = prompt.get(input1="This is real input", input2="This is real input 2")

    expected_prompt = """This is an instruction.
###
input1: This is an input.
input2: This is another input.
output: This is an output.
###
input1: This is real input
input2: This is real input 2"""

    assert prompt == expected_prompt
