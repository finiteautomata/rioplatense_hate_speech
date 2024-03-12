class FewShotPromptTemplate:
    def __init__(
        self,
        instruction,
        examples,
        input_variables,
        output_variables,
        separator,
        input_template=None,
        example_template=None,
    ):
        self.instruction = instruction
        self.input_variables = input_variables
        self.output_variables = output_variables

        self.example_template = example_template or "\n".join(
            [f"{var}: {{{var}}}" for var in input_variables + output_variables]
        )

        self.input_template = input_template or "\n".join(
            [
                f"{input_variable}: {{{input_variable}}}"
                for input_variable in input_variables
            ]
        )
        self.separator = separator

        ### Calculates "base" prompt
        prompt = self.instruction

        if examples:
            prompt += f"\n{self.separator}\n"
            prompt += f"\n{self.separator}\n".join(
                [self.example_template.format(**example) for example in examples]
            )
            prompt += f"\n{self.separator}\n"
        # Add input variables

        self.base_prompt = prompt

    def get(self, **kwargs):
        prompt = self.base_prompt

        if prompt[-1] != "\n":
            prompt += "\n"
        prompt += self.input_template.format(**kwargs)

        return prompt
