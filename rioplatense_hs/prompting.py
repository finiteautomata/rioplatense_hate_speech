from .preprocessing import preprocess_tweet
from .openai import get_completion
from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess

# Agregar Chain of thought
# Agregar que es español rioplatense
instruction = f"""Determinar si el siguiente mensaje contiene discurso discriminatorio o no. Entendemos que hay discurso discriminatorio en el tweet si tiene declaraciones de carácter intenso e irracional de rechazo, enemistad y aborrecimiento contra un individuo o contra un grupo, siendo estos objetivos de estas expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

- mujer: Mujeres
- lgbti: Género, identidad sexual
- racismo: Inmigrantes, extranjeros, religión, pueblos aborígenes u otras nacionalidades
- clase: Clase, situación socioeconómica o barrio de residencia
- discapacidad: Discapacidades, problemas de salud mental o de adicción al alcohol, drogas u otros estupefacientes
- política: Opinión o ideología política. Criticar al gobierno no cuenta como discurso discriminatorio
- aspecto: Aspecto, gordofobia o edad
- criminal: Antecedentes penales o estar privado de la libertad

En contexto tendremos un titular de una noticia y en texto el comentario a analizar. Responder una o varias de las características separadas por coma, o bien "nada" si no hay discurso discriminatorio. Pensá paso a paso la respuesta.
"""


examples = [
    {
        "contexto": "Al borde del llanto, Nati Jota pidió disculpas por sus repudiables tuits pero los justificó: 'Quería sumar seguidores'",
        "texto": "Show o nada mi amorrr!! La imbesilidad al palo!",
        "salida": "Es un insulto y no menciona nada relacionado a las características protegidas. La respuesta final es 'nada'",
    },
    {
        "contexto": "Violento motín en la cárcel de Devoto: tomaron e incendiaron pabellones",
        "texto": "Hermanito lo que son esos grones. Ahí adentro le robaron la billetera al coronavirus",
        "salida": 'La palabra "grones" significa negros, lo cual alude a su color de piel y aspecto. Además, se refiere a su condición de criminales. La respuesta final es "criminal, racismo, aspecto"',
    },
    {
        "contexto": 'Mayra Mendoza criticó el banderazo contra el Gobierno: "No tiene sentido y es antidemocrático"',
        "texto": "Que negrita villera....ni con la plata q roba puede arreglarse",
        "salida": 'El texto alude a su color de piel, a su condición social, y a su aspecto. La respuesta final es "mujer, racismo, aspecto, clase"',
    },
    {
        "contexto": "No da tregua: sin ropa y desafiante Flor Peña volvió a mostrar su lado más hot",
        "texto": "K lado hot tiene más varices k mi abuela. Tiene mas puesta de espalda k karagian trola",
        "salida": 'El texto alude a su aspecto (las varices) y la desprecia por "trola". La respuesta final es "mujer, aspecto"',
    },
    {
        "contexto": 'Wuhan celebra el fin de la cuarentena por el coronavirus con un mensaje para el resto del mundo: "Aprendan de nuestros errores"',
        "texto": "Chinos culiados de mierda ojalá se mueran todos",
        "salida": 'Desea que se mueran los chinos por ser los supuestos culpables del COVID-19. La respuesta final es "racismo"',
    },
    {
        "contexto": "Otro ataque de furia contra un colectivero: una mujer trans le pateó la unidad y le tiró un ladrillazo",
        "texto": "@usuario Un tipo operado. Con la fuerza de un hombre y no la de una mujer",
        "salida": 'El texto alude a que la mujer trans es un hombre. La respuesta final es "lgbti"',
    },
]

separator = "---"

# Join with ##

template_prompt = (
    instruction
    + f"\n{separator}\n"
    + f"\n{separator}\n".join(
        [
            f"contexto: {example['contexto']}\ntexto: {example['texto']}\nsalida: "
            for example in examples
        ]
    )
)


def build_prompt(contexto, texto):
    contexto = _pysent_preprocess(
        contexto,
        preprocess_hashtags=False,
        demoji=False,
        preprocess_handles=False,
    )
    texto = preprocess_tweet(texto)
    prompt = (
        template_prompt
        + f"\n{separator}\ncontexto: {contexto}\ntexto: {texto}\nsalida: "
    )

    return prompt


def get_response(contexto, texto, model="gpt-3.5-turbo"):
    """
    Get output from OpenAI API

    """
    prompt = build_prompt(contexto, texto)
    response = get_completion(prompt, model=model)
    text = response.choices[0].message.content

    return text
