from rioplatense_hs.preprocessing import label_to_text, labels, text_to_label


def test_empty_row_to_text():
    row = {l: 0 for l in labels}
    assert label_to_text(row) == "nada"


def test_single_label_row_to_text():
    row = {l: 0 for l in labels}

    row["WOMEN"] = 1

    assert label_to_text(row) == "mujer"


def test_multiple_labels():
    row = {l: 0 for l in labels}

    row["WOMEN"] = 1
    row["LGBTI"] = 1

    assert label_to_text(row) == "mujer, lgbti"


def test_three_labels():
    row = {l: 0 for l in labels}

    row["WOMEN"] = 1
    row["LGBTI"] = 1
    row["CLASS"] = 1

    assert label_to_text(row) == "mujer, lgbti, clase"


def test_discapacidad_and_apariencia():
    row = {l: 0 for l in labels}

    row["DISABLED"] = 1
    row["APPEARANCE"] = 1
    row["POLITICS"] = 1

    assert label_to_text(row) == "política, discapacidad, apariencia"


def test_text_to_label_nada_1():
    text = 'Esto es blablablablabla. La respuesta final es "nada".'

    empty_row = {l: 0 for l in labels}
    assert text_to_label(text) == empty_row


def test_tiene_mujer_pero_finalmente_no_es_nada():
    text = 'Esto es blablablablabla, dice algo de las mujeres. La respuesta final es "nada".'

    empty_row = {l: 0 for l in labels}

    assert text_to_label(text) == empty_row


def test_many_labels():
    text = 'Esto es blablablablabla, dice algo de las mujeres. La respuesta final es "clase, lgbti, racismo".'

    expected_row = {l: 0 for l in labels}

    expected_row["CLASS"] = 1
    expected_row["LGBTI"] = 1
    expected_row["RACISM"] = 1

    assert text_to_label(text) == expected_row


def test_it_is_case_insensitive():
    text = 'Esto es blablablablabla, dice algo de las mujeres. La respuesta final es "CLASE, lgbti, RACISMO".'

    expected_row = {l: 0 for l in labels}

    expected_row["CLASS"] = 1
    expected_row["LGBTI"] = 1
    expected_row["RACISM"] = 1

    assert text_to_label(text) == expected_row


def test_it_is_tilde_insensitive():
    text = 'Esto es blablablablabla, dice algo de las mujeres. La respuesta final es "CLÁSE, politica, RACÍSMO".'

    expected_row = {l: 0 for l in labels}

    expected_row["CLASS"] = 1
    expected_row["POLITICS"] = 1
    expected_row["RACISM"] = 1

    assert text_to_label(text) == expected_row
