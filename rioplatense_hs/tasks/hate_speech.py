import random
from ..preprocessing import preprocess_tweet
from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess
from ..prompting import FewShotPromptTemplate


# Agregar Chain of thought
# Agregar que es español rioplatense

# Determinar si el siguiente texto correspondiente a un tweet y presentado con un contexto, contiene discurso de odio y contra qué característica protejida. Entendemos que hay discurso de odio en el tweet si tiene declaraciones de carácter intenso y/o irracional de rechazo, enemistad y aborrecimiento contra un individuo o contra un grupo, siendo el individuo o el grupo objetivo de estas expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

instruction = f"""Determinar si el siguiente texto, correspondiente a un tweet y presentado con un contexto, contiene discurso de odio. Entendemos que hay discurso de odio si tiene declaraciones de carácter intenso y/o irracional de rechazo, enemistad o aborrecimiento contra un individuo o contra un grupo, siendo el individuo o el grupo objetivo de las expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

- mujer: hace referencia a mujeres o al movimiento feminista.
- lgbti: hace referencia a gays, lesbianas, transexuales y otras identidades de género.
- racismo: hace referencia a inmigrantes, xenofobia, o a pueblos aborígenes.
- clase: hace referencia a personas de bajos recursos o motivos de clase.

Los tweets están escritos en español rioplatense, y dentro del contexto cultural de Argentina. Responder una o varias de las características separadas por coma, o bien "nada" si no hay discurso de odio. Pensar y justificar paso a paso la respuesta antes de responder."""


examples = [
    {
        "contexto": 'Wuhan celebra el fin de la cuarentena por el coronavirus con un mensaje para el resto del mundo: "Aprendan de nuestros errores"',
        "texto": "Chinos culiados de mierda ojalá se mueran todos",
        "salida": 'El texto insulta a los chinos ("culiados") y desea que se mueran, implicando que son los culpables del COVID-19. La respuesta final es "racismo".',
        "labels": ["RACISM"],
    },
    {
        "contexto": "Al borde del llanto, Nati Jota pidió disculpas por sus repudiables tuits pero los justificó: 'Quería sumar seguidores'",
        "texto": "Show o nada mi amorrr!! Sos una imbesil!",
        "salida": 'El texto contiene un insulto pero no menciona nada relacionado a las características protegidas, pese a mencionar a una mujer. La respuesta final es "nada".',
        "labels": [],
    },
    {
        "contexto": 'Funcionario público y drag queen: la "doble vida" del político que representa a Salta en tacos, peluca y barba con brillos',
        "texto": "Que asco..!'",
        "salida": 'El texto expresa desagrado hacia la identidad de género de la persona. La respuesta final es "lgbti".',
        "labels": ["LGBTI"],
    },
    {
        "contexto": "Violento motín en la cárcel de Devoto: tomaron e incendiaron pabellones",
        "texto": "Hermanito lo que son esos grones. Ahí adentro le robaron la billetera al coronavirus",
        "salida": 'El texto utiliza la palabra "grones", que significa negros (se obtiene "grones" a partir de "negros" cambiando el orden de las sílabas). Negro o grone es una manera despectiva de refererirse a alguien por su color de piel. Además, da a entender que los presos son ladrones. La respuesta final es "racismo".',
        "labels": ["RACISM"],
    },
    {
        "contexto": 'Mayra Mendoza criticó el banderazo contra el Gobierno: "No tiene sentido y es antidemocrático"',
        "texto": "Que negrita villera....ni con la plata q roba puede arreglarse",
        "salida": 'El texto alude al color de piel de Mayra Mendoza ("negrita") y a su condición social ("villera") de manera discriminatoria. La respuesta final es "racismo, clase".',
        "labels": ["RACISM", "CLASS"],
    },
    {
        "contexto": "No da tregua: sin ropa y desafiante Flor Peña volvió a mostrar su lado más hot",
        "texto": "K lado hot tiene más varices k mi abuela. Tiene mas puesta de espalda k karagian trola",
        "salida": 'El texto alude al aspecto de Flor Peña (sus várices) y la desprecia por "trola", un insulto que significa "prostituta". La respuesta final es "mujer".',
        "labels": ["WOMEN"],
    },
    {
        "contexto": "Otro ataque de furia contra un colectivero: una mujer trans le pateó la unidad y le tiró un ladrillazo",
        "texto": "Un tipo operado. Con la fuerza de un hombre y no la de una mujer",
        "salida": 'El texto asegura que la mujer trans es "un tipo operado", implicando que en los hechos es un hombre y negando su identidad de género. La respuesta final es "lgbti"',
        "labels": ["LGBTI"],
    },
    {
        "contexto": 'Elisa Carrió denunció que el Gobierno usa la pandemia para "establecer un estado de sitio"',
        "texto": "Gorda psiquiátrica",
        "salida": 'El texto habla de su aspecto (gorda) y la acusa de tener problemas psiquiátricos. Sin embargo, no hace ninguna alusión a una característica protegida. La respuesta final es "nada".',
        "labels": [],
    },
    {
        "contexto": 'Loly Antoniale mostró su impresionante casa en Miami: "Soy la reina de mi castillo"',
        "texto": "No pudo enganchar al viejo famoso..se busco otro..y este le puso su castillo en Miami...",
        "salida": 'El texto alude a la vida amorosa de Loly Antoniale, sugiriendo que busca hombres por interés económico. Esta apreciación es sexista y despectiva. La respuesta final es "mujer".',
        "labels": ["WOMEN"],
    },
    {
        "contexto": "Les darán DNI provisorio a personas en situación vulnerable, para que puedan empezar a cobrar planes sociales",
        "texto": "Seguimos alimentando vagos",
        "salida": 'El texto se refiere a quienes cobran planes sociales como vagos a quienes hay que alimentar. Esto muestra un desprecio hacia las clases bajas que requieren de la asistencia estatal. La respuesta final es "clase".',
        "labels": ["CLASS"],
    },
    {
        "contexto": "Los dos presos heridos de bala en el motín de Devoto tienen Covid-19 y uno quedó hemipléjico",
        "texto": "justicia divina!",
        "salida": "El comentario contiene una celebración frente a un ataque a presos. A pesar de ser sumamente ofensivo, no hace ninguna alusión a una característica protegida. La respuesta final es 'nada'.",
        "labels": [],
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

    input_template = """contexto: {contexto}
texto: {texto}
salida:
"""

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["contexto", "texto"],
        output_variables=["salida"],
        examples=used_examples,
        separator=separator,
        input_template=input_template,
        **kwargs,
    )

    return prompt.get(contexto=contexto, texto=texto)
