"""
Checks the PDF question answering against questions whose answers are known
(from Allowances.pdf). Needs Ollama running. Run it after any change:

    python eval_pdf.py          # show only failures
    python eval_pdf.py -v       # show every answer

Each case lists patterns the answer must contain (all of them) and patterns
it must not contain. Replace or extend CASES with your own questions.
"""
import re
import sys

from document_qa import answer_pdf_question
from document_search import ensure_pdf_index

# (question, must match all, must match none)
CASES = [
    ("What is the overtime allowance policy?",
     [r"basic pay", r"departmental head|prior approval"], []),

    ("What is the food allowance?",
     [r"\b200\b", r"\b400\b", r"actual"],
     [r"6,000", r"8,000", r"20,000", r"12,000"]),

    ("what is the overtime policy for grade 6 employees only?",
     [r"direct sales", r"\bnot\b|n't"], []),

    ("What is the hotel stay allowance for grade 6?",
     [r"8,000"], [r"20,000", r"12,000"]),

    ("What is the hotel limit for grade 12?",
     [r"12,000"], [r"20,000"]),

    ("What is the guest house allowance for grade 14?",
     [r"6,000"], [r"12,000"]),

    ("How much is the food allowance for grade 5?",
     [r"\b400\b"], [r"\b200\b", r"actual"]),

    ("What is the food allowance for grade 11?",
     [r"actual"], [r"\b400\b", r"\b200\b"]),

    ("Is overtime paid during foreign travel?",
     [r"\bno\b|\bnot\b", r"tour|travel"], []),

    ("How many days can I claim daily allowance at one station?",
     [r"\b30\b"], []),

    ("What is the disturbance allowance?",
     [r"one month", r"gross salary"], []),

    ("Who pays for excess baggage on foreign travel?",
     [r"employee"], []),

    ("Tell me about the mobile phone allowance",
     [r"discretion|manager", r"telephone directory"], []),

    ("What does transfer cover for the family?",
     [r"spouse", r"children"], []),

    # ---- Holiday & Leave .pdf
    ("What is the difference between the types of leave?",
     [r"privilege", r"casual", r"sick", r"maternity"], []),

    ("How many days of casual leave do I get?",
     [r"\b10\b"], []),

    ("What is the sick leave policy?",
     [r"\b14\b", r"medical certificate"], []),

    ("What is the working hours calculation for AESL?",
     [r"2,?400"], [r"3,?000"]),

    ("What is the working hour calculation for AML?",
     [r"2,?350", r"\b300\b", r"2,?400"], []),

    ("What about total working hours at AESL or ARM and its details?",
     [r"2,?400", r"2,?650"], []),

    # The PDF says nothing about weekends: the answer should say so.
    ("If a public holiday falls on a weekend, do we get another day off?",
     [r"(not|n't|no)\s+(say|state|mention|specif|cover|address|information)"
      r"|doesn't|does not"], []),

    ("Mobile phone allowance details",
     [r"discretion|manager", r"telephone directory"], []),

    ("How much allowance do I get for food per hour?",
     [r"\b200\b", r"\b400\b"], []),

    ("Who do I report attendance errors to?",
     [r"rafay ansar"], []),

    # ---- Cross Functional Teams.pdf (lists split across page breaks)
    ("Cross Functional Team Lahore",
     [r"nasim iqbal", r"aamir shaiq", r"rashid jahangri", r"farooq ijaz"],
     [r"abid hussain", r"saifullah", r"hasan saeed"]),

    ("Cross Functional Team Islamabad",
     [r"iftikhar wajid", r"hammad umer", r"usman ahmed"],
     [r"farooq", r"saifullah"]),

    ("Allied Blood Donation Team",
     [r"hasan saeed", r"aqeel ahmed khan", r"abid hussain", r"muhammed afzal",
      r"saifullah"],
     [r"daniyal", r"waleed"]),

    ("Who is in the Sports Team?",
     [r"daniyal khan", r"waleed malik", r"hasnain"],
     [r"hasan saeed", r"abid hussain"]),

    ("Cross functional teams document summary in detail",
     [r"karachi", r"lahore", r"islamabad", r"transport", r"food", r"sports",
      r"blood donation", r"farooq ijaz", r"saifullah", r"fortnight"],
     [r"not specified"]),

    # Needs the embedding model: "extra hours" never appears in the PDF.
    ("What will be the extra hours pay?",
     [r"basic pay"], []),
]

# Questions the PDFs cannot answer: the bot should say it found nothing.
NOT_IN_PDF = [
    "What is the capital of France?",
]


def main():
    verbose = "-v" in sys.argv
    ensure_pdf_index()
    failures = 0

    for question, must, must_not in CASES:
        result = answer_pdf_question(question)
        answer = result["answer"]

        missing = [p for p in must if not re.search(p, answer, re.I)]
        forbidden = [p for p in must_not if re.search(p, answer, re.I)]
        passed = not missing and not forbidden and result.get("found", True)
        failures += not passed

        print(f"{'PASS' if passed else 'FAIL'}  {question}")

        if not passed:
            if missing:
                print(f"      answer should mention: {missing}")
            if forbidden:
                print(f"      answer should not mention: {forbidden}")

        if verbose or not passed:
            print("      " + answer.replace("\n", "\n      "))

    for question in NOT_IN_PDF:
        result = answer_pdf_question(question)
        passed = result.get("found") is False
        failures += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {question}  (should find nothing)")

        if verbose or not passed:
            print("      " + result["answer"].replace("\n", "\n      "))

    # Sources should carry the revision date printed in the PDF.
    sources = answer_pdf_question("What is the food allowance?")["sources"]
    passed = any(s.get("last_modified") for s in sources)
    failures += not passed
    print(f"{'PASS' if passed else 'FAIL'}  sources include the PDF's 'Last Modified' date  {sources[:1]}")

    total = len(CASES) + len(NOT_IN_PDF) + 1
    print(f"\n{total - failures}/{total} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()