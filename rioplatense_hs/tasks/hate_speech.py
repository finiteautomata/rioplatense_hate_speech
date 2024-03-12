import random
from ..preprocessing import preprocess_tweet
from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess
from ..prompting import FewShotPromptTemplate


# Agregar Chain of thought
# Agregar que es español rioplatense
instruction = f"""Determinar si el siguiente mensaje contiene discurso de odio. Entendemos que hay discurso de odio en el tweet si tiene declaraciones de carácter intenso e irracional de rechazo, enemistad y aborrecimiento contra un individuo o contra un grupo, siendo objetivos de estas expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

- mujer: mujeres o movimiento feminista
- lgbti: contra gays, lesbianas, transexuales y otras identidades de género
- racismo: inmigrantes, xenofobia, o contra pueblos aborígenes
- clase: personas de bajos recursos o motivos de clase

Responder una o varias de las características separadas por coma, o bien "nada" si no hay discurso de odio. Pensá paso a paso la respuesta antes de responder."""


examples = [
    {
        "contexto": 'Wuhan celebra el fin de la cuarentena por el coronavirus con un mensaje para el resto del mundo: "Aprendan de nuestros errores"',
        "texto": "Chinos culiados de mierda ojalá se mueran todos",
        "salida": 'Desea que se mueran los chinos por ser los supuestos culpables del COVID-19. La respuesta final es "racismo".',
        "labels": ["RACISM"],
    },
    {
        "contexto": "Al borde del llanto, Nati Jota pidió disculpas por sus repudiables tuits pero los justificó: 'Quería sumar seguidores'",
        "texto": "Show o nada mi amorrr!! Sos una imbesil!",
        "salida": 'Es un insulto pero no menciona nada relacionado a las características protegidas. La respuesta final es "nada".',
        "labels": [],
    },
    {
        "contexto": "Violento motín en la cárcel de Devoto: tomaron e incendiaron pabellones",
        "texto": "Hermanito lo que son esos grones. Ahí adentro le robaron la billetera al coronavirus",
        "salida": 'La palabra "grones" significa negros al revés, lo cual alude a su color de piel. La respuesta final es "racismo".',
        "labels": ["RACISM"],
    },
    {
        "contexto": 'Mayra Mendoza criticó el banderazo contra el Gobierno: "No tiene sentido y es antidemocrático"',
        "texto": "Que negrita villera....ni con la plata q roba puede arreglarse",
        "salida": 'El texto alude a su color de piel y a su condición social (villera). La respuesta final es "racismo, clase".',
        "labels": ["RACISM", "CLASS"],
    },
    {
        "contexto": "No da tregua: sin ropa y desafiante Flor Peña volvió a mostrar su lado más hot",
        "texto": "K lado hot tiene más varices k mi abuela. Tiene mas puesta de espalda k karagian trola",
        "salida": 'El texto alude a su aspecto (las varices) y la desprecia por "trola", un insulto que significa "prostituta". La respuesta final es "mujer".',
        "labels": ["WOMEN"],
    },
    {
        "contexto": "Otro ataque de furia contra un colectivero: una mujer trans le pateó la unidad y le tiró un ladrillazo",
        "texto": "Un tipo operado. Con la fuerza de un hombre y no la de una mujer",
        "salida": 'El texto alude a que la mujer trans es un hombre. La respuesta final es "lgbti"',
        "labels": ["LGBTI"],
    },
    {
        "contexto": 'Elisa Carrió denunció que el Gobierno usa la pandemia para "establecer un estado de sitio"',
        "texto": "Gorda psiquiátrica",
        "salida": 'El texto alude a su aspecto (gorda) y la acusa de tener problemas psiquiátricos. Sin embargo, no hace ninguna alusión a una característica protegida. La respuesta final es "nada".',
        "labels": [],
    },
    {
        "contexto": "Les darán DNI provisorio a personas en situación vulnerable, para que puedan empezar a cobrar planes sociales",
        "texto": "Seguimos alimentando vagos",
        "salida": 'El comentario se refiere a quienes cobran planes sociales como vagos. La respuesta final es "clase".',
        "labels": ["CLASS"],
    },
    {
        "contexto": "'País de maricas': ¿Por qué Jair Bolsonaro vuelve a atacar a los homosexuales en Brasil?",
        "texto": 'El discurso de bolsonaro donde dice "maricas" es brillante, sólo eso basta para meresca ser el presidente de Sudamérica. Clarín es una cueva de zurdos y progresistas que se pintan los labios punto. Clarín suavecito, te pasas de marica!',
        "salida": 'El texto alude a la homosexualidad como algo negativo, a la vez que acusa de ser "zurdos y progresistas" a quienes no están de acuerdo con el discurso de Bolsonaro. La respuesta final es "lgbti"',
        "labels": ["LGBTI"],
    },
    # {
    #     "texto": "Como se llama la obra? Estigmaticemos a los chinos",
    #     "contexto": "Coronavirus: las terribles imágenes del mercado donde se originó la pandemia",
    #     "salida": 'El texto critica el titular del artículo por "estigmatizar a los chinos". Es un comentario irónico. La respuesta final es "nada"',
    #     "labels": [],
    # },
    {
        "contexto": "Los dos presos heridos de bala en el motín de Devoto tienen Covid-19 y uno quedó hemipléjico",
        "texto": "justicia divina!",
        "salida": "El texto alude a que los presos merecen ser baleados. Sin embargo, no hace ninguna alusión a una característica protegida. La respuesta final es 'nada'",
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

    prompt = FewShotPromptTemplate(
        instruction=instruction,
        input_variables=["contexto", "texto"],
        output_variables=["salida"],
        examples=used_examples,
        separator=separator,
        **kwargs,
    )

    return prompt.get(contexto=contexto, texto=texto)
