import json

questions = []
answers = []

with open("data/qa/generated_qa.jsonl") as f:
    for line in f:
        item = json.loads(line)
        questions.append(item["question"])
        answers.append(item["answer"])

with open("data/eval/questions.txt", "w") as qf:
    for q in questions:
        qf.write(q + "\n")

with open("data/eval/answers_gold.txt", "w") as af:
    for a in answers:
        af.write(a + "\n")