import algoris

while True:
    text = input('algoris > ')
    result, error = algoris.run('<stdin>', text)

    if error: print(error.as_string())
    elif result: print(repr(result))