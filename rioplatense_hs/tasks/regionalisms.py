import random
from ..prompting import FewShotPromptTemplate
from ..preprocessing import preprocess_tweet
from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess


# Let's build a prompt to ask ChatGPT if there is a regionally specific term or expression that is not clear to it

instruction = f"""Explicar si existe un término o expresión regional del español de Argentina en el texto a continuación. Un regionalismo es una expresión de carácter usualmente coloquial usada en alguna variedad dialectal del español. No consideramos nombres propios (personas, lugares, etc) como tales. La expresión regional puede estar en el contexto (titular de noticia) o en el comentario. Razonar paso a paso antes de dar la respuesta."""


examples = [
    {
        "contexto": "¿More Rial cuestionó las habilidades de su ex en la cama?Luego de su separación con Facundo Ambrosioni, la mediática hizo una fuerte referencia a su vida sexual",
        "texto": "Esta acumulando mas leche que mastellone",
        "respuesta": 'La expresión "acumulando mas leche" es de carácter regional, y se refiere a la acumulación de deseos sexuales. La respuesta final es sí.',
    },
    {
        "contexto": "Enrique Pinti no descarta una nueva postulación de Macri: “En este país siempre hay una chance para cualquiera que nos vuelva a cagar”",
        "texto": "Que bajo a caido, PINTI, ese viejo TROLO SIGUE CON EL CURRO DE HACER HUMOR POLITICO.",
        "respuesta": 'La expresión "trolo" es un término regional en Argentina usado para referirse a una persona homosexual. La respuesta final es sí.',
    },
    {
        "contexto": "Coronavirus: En medio de la pandemia, más de 20.000 chinos se agolparon en una turística montaña",
        "texto": "Cuando China despierte! La amenaza amarilla!! Decían los abuelos y pasó! Ojalá no se les vuelva a salir de las manos!!",
        "respuesta": "El comentario no contiene expresiones regionales. La respuesta final es no.",
    },
]


separator = "###"


def build_prompt(contexto, texto, num_examples=None, shuffle=False, seed=42, **kwargs):
    """
    Builds HS prompt

    Args:
        contexto (str): Context tweet
        texto (str): Text to predict
        num_examples (int): Number of examples to use
        shuffle (bool): Whether to shuffle examples
        seed (int): Random seed
        kwargs: Additional arguments to pass to FewShotPromptTemplate
    """

    contexto = _pysent_preprocess(
        contexto,
        preprocess_hashtags=False,
        demoji=False,
        preprocess_handles=False,
    )
    texto = preprocess_tweet(texto)

    if num_examples is None:
        num_examples = len(examples)

    if shuffle:
        random.seed(seed)
        used_examples = random.sample(examples, num_examples)
    else:
        used_examples = examples[:num_examples]

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["contexto", "texto"],
        output_variables=["respuesta"],
        examples=used_examples,
        separator=separator,
        **kwargs,
    )

    return prompt.get(contexto=contexto, texto=texto)
