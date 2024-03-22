import re
from .tasks.hate_speech import instruction, examples

instruction_template = """[INST] {instruction}
contexto: {contexto}
texto: {texto}
[/INST]"""


first_instruction = instruction

second_instruction = """Determinar si el siguiente mensaje contiene discurso de odio."""


instruction_template = """[INST] {instruction}
contexto: {contexto}
texto: {texto}
[/INST]"""

example_template = (
    instruction_template
    + """
{output}
</s>
"""
)


def get_prompt(context, text):
    # Uso algo distinto acá
    first_example = examples[0]
    prompt = example_template.format(
        instruction=first_instruction,
        contexto=first_example["contexto"],
        texto=first_example["texto"],
        output=first_example["salida"],
    )

    # Add next examples with second_instruction

    for example in examples[1:]:
        prompt += example_template.format(
            instruction="",
            contexto=example["contexto"],
            texto=example["texto"],
            output=example["salida"],
        )
    prompt += instruction_template.format(
        instruction="",
        contexto=context,
        texto=text,
    )

    return prompt


answer_regex = r"la respuesta final es (['\"]).*?(\1)"


def post_process_output(text):
    """
    Post-process the output to remove the rest of the text after the answer

    Args:

    text (str): Text to post-process

    Returns:

    str: Post-processed text
    """
    match = re.search(answer_regex, text, flags=re.IGNORECASE)
    if not match:
        return text
    # Busco que termine la actual oración

    end = match.end()

    while end < len(text) and text[end] not in [".", "!", "?"]:
        end += 1

    return text[: end + 1]
