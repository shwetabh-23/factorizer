from factorizer import Factorizer

tokenizer = Factorizer("sym_en_256_final_vocab.dawg")
tokenizer.sigma = 0.01

sentence = "the echo of a distant time comes willowing across the sand, and everything is green and submarine܂׀Ԅ"

encoding = tokenizer(sentence)

print(f"INPUT:    {sentence}")
print(f"SUBWORDS: {' '.join(encoding.tokens)}") # not breaking down the sentence into subwords
print(f"INDICES:  {' '.join(str(index) for index in encoding.ids)}")
english = Factorizer("/nas/ckgfs/users/thawani/etok/english.dawg") # ܂׀ԅRhymes܂׀Ԅ
print(f"DECODED:  {tokenizer.decode(encoding.ids)}")