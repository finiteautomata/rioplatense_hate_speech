from rioplatense_hs.prompting import FewShotPromptTemplate


def test_generate_template_with_no_examples():
    instruction = "This is an instruction."

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["input"],
        output_variables=["output"],
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
