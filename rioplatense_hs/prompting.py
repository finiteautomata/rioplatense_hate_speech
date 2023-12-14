import random
from .preprocessing import preprocess_tweet
from .openai import get_completion
from pysentimiento.preprocessing import preprocess_tweet as _pysent_preprocess

# Agregar Chain of thought
# Agregar que es español rioplatense
instruction = f"""Determinar si el siguiente mensaje contiene discurso de odio. Entendemos que hay discurso de odio en el tweet si tiene declaraciones de carácter intenso e irracional de rechazo, enemistad y aborrecimiento contra un individuo o contra un grupo, siendo objetivos de estas expresiones por poseer una característica protegida. Las características protegidas que contemplamos son:

- mujer: mujeres o movimiento feminista
- lgbti: contra gays, lesbianas, transexuales y otras identidades de género
- racismo: inmigrantes, xenofobia, o contra pueblos aborígenes
- clase: personas de bajos recursos o motivos de clase
- discapacidad: discapacidad, problemas de salud mental o de adicción al alcohol u otras drogas
- política: ideología política
- aspecto: aspecto, gordofobia o edad
- criminal: presos o delincuentes comunes

Responder una o varias de las características separadas por coma, o bien "nada" si no hay discurso de odio. Pensá paso a paso la respuesta antes de responder."""


examples = [
    {
        "contexto": 'Wuhan celebra el fin de la cuarentena por el coronavirus con un mensaje para el resto del mundo: "Aprendan de nuestros errores"',
        "texto": "Chinos culiados de mierda ojalá se mueran todos",
        "salida": 'Desea que se mueran los chinos por ser los supuestos culpables del COVID-19. La respuesta final es "racismo"',
    },
    {
        "contexto": "Al borde del llanto, Nati Jota pidió disculpas por sus repudiables tuits pero los justificó: 'Quería sumar seguidores'",
        "texto": "Show o nada mi amorrr!! Sos una imbesil!",
        "salida": "Es un insulto y no menciona nada relacionado a las características protegidas. La respuesta final es 'nada'",
        "labels": [],
    },
    {
        "contexto": "Violento motín en la cárcel de Devoto: tomaron e incendiaron pabellones",
        "texto": "Hermanito lo que son esos grones. Ahí adentro le robaron la billetera al coronavirus",
        "salida": 'La palabra "grones" significa negros, lo cual alude a su color de piel y aspecto. Además, se refiere a su condición de presos. La respuesta final es "criminal, racismo, aspecto"',
        "labels": ["CRIMINAL", "RACISM", "APPEARANCE"],
    },
    {
        "contexto": "Pileta, bar y muelle: Florencia Peña abrió las puertas de su lujosa nueva casa",
        "texto": "pobrecita no tiene para comer",
        "salida": "El comentario se refiere irónicamente a la lujosa casa, pero sin hacer ninguna mención a característica protegida. La respuesta final es 'nada'.",
        "labels": [],
    },
    {
        "contexto": 'Mayra Mendoza criticó el banderazo contra el Gobierno: "No tiene sentido y es antidemocrático"',
        "texto": "Que negrita villera....ni con la plata q roba puede arreglarse",
        "salida": 'El texto alude a su color de piel, a su condición social (villera), y a su aspecto. La respuesta final es "racismo, aspecto, clase"',
        "labels": ["RACISM", "CLASS", "APPEARANCE"],
    },
    {
        "contexto": "Les darán DNI provisorio a personas en situación vulnerable, para que puedan empezar a cobrar planes sociales",
        "texto": "Seguimos alimentando vagos",
        "salida": 'El comentario se refiere a quienes cobran planes sociales como vagos. La respuesta final es "clase"',
        "labels": ["CLASS"],
    },
    {
        "contexto": "Viola Davis, Halle Berry y Angela Bassett, actrices negras que superaron la discriminación en Hollywood",
        "texto": "el titulo de por si da pena, la nación cree que el racismo es una enfermedad que los que la sufren tienen que superar, es un título discriminatorio",
        "salida": 'El comentario critica el título del artículo por insinuar que el racismo es una enfermedad. No critica en ningún momento a las actrices. La respuesta final es "nada"',
        "labels": [],
    },
    {
        "contexto": "No da tregua: sin ropa y desafiante Flor Peña volvió a mostrar su lado más hot",
        "texto": "K lado hot tiene más varices k mi abuela. Tiene mas puesta de espalda k karagian trola",
        "salida": 'El texto alude a su aspecto (las varices) y la desprecia por "trola". La respuesta final es "mujer, aspecto"',
        "labels": ["WOMEN", "APPEARANCE"],
    },
    {
        "contexto": 'Ginés González García: "Le tengo mucho miedo al verano, el riesgo va a seguir"',
        "texto": "A este viejo le quedan 2 neuronas 1 para comer 2 para cagar",
        "salida": 'El texto alude a su edad, aduciendo que no tiene capacidades mentales acordes. La respuesta final es "aspecto"',
        "labels": ["APPEARANCE"],
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
        "salida": 'El texto alude a su aspecto (gorda) y la acusa de tener problemas psiquiátricos. La respuesta final es "aspecto, discapacidad"',
        "labels": ["APPEARANCE", "DISABLED"],
    },
    {
        "contexto": "'País de maricas': ¿Por qué Jair Bolsonaro vuelve a atacar a los homosexuales en Brasil?",
        "texto": 'El discurso de bolsonaro donde dice "maricas" es brillante, sólo eso basta para meresca ser el presidente de Sudamérica. Clarín es una cueva de zurdos y progresistas que se pintan los labios punto. Clarín suavecito, te pasas de marica!',
        "salida": 'El texto alude a la homosexualidad como algo negativo, a la vez que acusa de ser "zurdos y progresistas" a quienes no están de acuerdo con el discurso de Bolsonaro. La respuesta final es "lgbti, política"',
        "labels": ["LGBTI", "POLITICS"],
    },
    {
        "texto": "Como se llama la obra? Estigmaticemos a los chinos",
        "contexto": "Coronavirus: las terribles imágenes del mercado donde se originó la pandemia",
        "salida": 'El texto critica el titular del artículo por "estigmatizar a los chinos". Es un comentario irónico. La respuesta final es "nada"',
        "labels": [],
    },
    {
        "contexto": "Los dos presos heridos de bala en el motín de Devoto tienen Covid-19 y uno quedó hemipléjico",
        "texto": "justicia divina!",
        "salida": "El texto alude a que los presos merecen ser baleados. La respuesta final es 'criminal'",
    },
]

separator = "###"


def build_base_prompt(num_examples=None, shuffle=False, seed=42):
    if num_examples is None:
        num_examples = len(examples)

    if shuffle:
        random.seed(seed)
        used_examples = random.sample(examples, num_examples)
    else:
        used_examples = examples[:num_examples]

    template_prompt = (
        instruction
        + f"\n{separator}\n"
        + f"\n{separator}\n".join(
            [
                f"contexto: {example['contexto']}\ntexto: {example['texto']}\nsalida: {example['salida']} "
                for example in used_examples
            ]
        )
    )

    return template_prompt


def build_prompt(contexto, texto, base_prompt):
    """
    Builds prompt for OpenAI API

    Args:
        contexto (str): Context tweet
        texto (str): Text to predict
        base_prompt (str): Base prompt to use
    """
    contexto = _pysent_preprocess(
        contexto,
        preprocess_hashtags=False,
        demoji=False,
        preprocess_handles=False,
    )
    texto = preprocess_tweet(texto)
    prompt = (
        base_prompt + f"\n{separator}\ncontexto: {contexto}\ntexto: {texto}\nsalida: "
    )

    return prompt


def get_response(
    contexto,
    texto,
    base_prompt,
    model="gpt-3.5-turbo",
):
    """
    Get output from OpenAI API

    """
    prompt = build_prompt(contexto, texto, base_prompt=base_prompt)
    response = get_completion(prompt, model=model)
    text = response.choices[0].message.content

    return prompt, text
