from defsent import DefSent

def main():
    model = DefSent("cl-nagoya/defsent-bert-base-uncased-cls")
    print("please input any sentences!")
    while True:
        sentence = input("> ")
        [words] = model.predict_words(sentence)
        line = "  ".join(words)
        print(f"predicted:  {line}")

if __name__ == "__main__":
    main()