from hstest.stage_test import StageTest
from hstest.test_case import TestCase
from hstest.check_result import CheckResult

import re


# function to provide better feedback
def get_lines_with_key_words(lines, keywords):
    lines_with_keywords = []
    for line in lines:
        if set(line.lower().split()) & set(keywords):
            lines_with_keywords.append(line)
    return lines_with_keywords


class FakeNewsTest(StageTest):

    def generate(self):
        return [TestCase(stdin= [self.check],time_limit=12000000000)]
    def check(self, reply, attach):
        lines = reply.split('\n')
        if "" in lines:
            lines = list(filter(lambda a: a != "", lines))


        relevant_lines = get_lines_with_key_words(lines, keywords=['model'])

        # general
        if len(relevant_lines) != 1:
            return CheckResult.wrong(
                feedback=f"Expected 1 line with Accuracy, found {len(relevant_lines)}\n")

        accuracy_reply = re.findall(r'\d*\.\d+|\d+', relevant_lines[0])
        if len(accuracy_reply) != 1:
            return CheckResult.wrong(feedback=f'It should be one number in the "Accuracy:" section')
        # 1% error rate is allowed, right accuracy = 0.812
        if not 0.98 * 0.865 < float(accuracy_reply[0]) < 1.02 * 0.865:
            return CheckResult.wrong(feedback=f"Wrong accuracy")

        return CheckResult.correct()

if __name__ == '__main__':
    FakeNewsTest().run_tests()