import fire
import requests
import asyncio
import aiohttp
import json
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


async def get_definition(definition_dict):
    link = definition_dict["link"]

    # Async sleep to avoid getting banned
    await asyncio.sleep(0.25)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(link) as response:
                soup = BeautifulSoup(await response.text(), "html.parser")
    except Exception as e:
        print(f"Error getting definition for {definition_dict['term']} -- {link}")
        print(e)
        return definition_dict

    # Get definition

    panels = soup.find_all("div", class_="panel")

    definitions = []

    for panel in panels:
        pars = panel.find_all("p")

        if not pars:
            continue

        definition = pars[0].text.strip()
        examples = [p.text.strip() for p in pars[1:-1]]

        definitions.append(
            {
                "definition": definition,
                "examples": examples,
            }
        )

    return {
        **definition_dict,
        "definitions": definitions,
    }


async def get_terms_from_page(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            soup = BeautifulSoup(await response.text(), "html.parser")

    terms = soup.find_all("ul", class_="terms")
    terms = terms[0]

    ret = {}

    for term in tqdm(terms.find_all("li")):
        # Si tiene clase title, saltar
        klass = term.get("class")
        if klass and "title" in klass[0]:
            continue
        text = term.text

        if "(" in text:
            # Saco todo lo del paréntesis hasta el final
            text = text.split("(")[0]
        text = text.strip().lower()

        link = term.find("a")
        href = link.get("href") if link else ""
        ret[text] = {"term": text, "link": href}

    funs = [get_definition(v) for v in ret.values()]

    for f in tqdm(asyncio.as_completed(funs), total=len(funs)):
        definition_dict = await f

        ret[definition_dict["term"]] = definition_dict

    return ret


async def main(output_path):
    urls = [
        "https://www.diccionarioargentino.com/terms/" + l
        for l in "abcdefghijklmnopqrstuvwxyz"
    ]

    # tqdm asyncio loop for progress bar

    ret = {}

    for url in tqdm(urls):
        ret.update(await get_terms_from_page(url))

    print(f"Writing to file {len(ret)} expressions")
    with open(output_path, "w") as f:
        json.dump(ret, f, indent=4)


def scrape_diccionario(output):
    print(f"Scraping diccionario argentino to {output}")

    # Launch async loop

    asyncio.run(main(output))


if __name__ == "__main__":
    fire.Fire(scrape_diccionario)
