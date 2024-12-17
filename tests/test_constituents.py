from divide_and_conquer_sentiment.constituents import ABSASubpredictor


def test_dupa():
    a = ABSASubpredictor.from_pretrained(
        "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
        "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
        spacy_model="en_core_web_lg",
    )

    a.predict(["Behold", "Behold", "The food was great but the service was terrible"])

    print("awd")
