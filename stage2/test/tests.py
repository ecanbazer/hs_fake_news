from hstest.stage_test import StageTest
from hstest.test_case import TestCase
from hstest.check_result import CheckResult

import re


# function to provide better feedback
def get_model_name(line_reply):
    idx = line_reply.replace(" ", "").lower().index('model:') + len('model:')
    model_name_reply = line_reply.replace(" ", "")[idx:]
    return model_name_reply

def get_lines_with_key_words(lines, keywords):
    lines_with_keywords = []
    for line in lines:
        if set(line.lower().split()) & set(keywords):
            lines_with_keywords.append(line)
    return lines_with_keywords


class FakeNewsTest(StageTest):

    def generate(self):
        return [TestCase(stdin= [self.check],time_limit=120000)]
    def check(self, reply, attach):
        lines = reply.split('\n')
        if "" in lines:
            lines = list(filter(lambda a: a != "", lines))


        relevant_lines = get_lines_with_key_words(lines, keywords=['model:', 'accuracy:'])

        # general
        if len(relevant_lines) != 4:
            return CheckResult.wrong(
                feedback=f"Expected 4 lines with Model:/Accuracy:, found {len(relevant_lines)}\n"
                         f"Note that the order of the models in the output is important (see the Example section)")

        # models and accuracies print
        # 1st model
        model_name_answer = 'SVC'
        if model_name_answer not in relevant_lines[0]:
            model_name_reply = get_model_name(relevant_lines[0])
            return CheckResult.wrong(feedback=f"Incorrect name of the 1st model\n"
                                              f"Expected {model_name_answer}, found {model_name_reply}")
    

        accuracy_reply = re.findall(r'\d*\.\d+|\d+', relevant_lines[1])
        if len(accuracy_reply) != 1:
            return CheckResult.wrong(feedback=f'It should be one number in the "Accuracy:" section')
        # 1% error rate is allowed, right accuracy = 0.854
        if not 0.99 * 0.854 < float(accuracy_reply[0]) < 1.01 * 0.854:
            return CheckResult.wrong(feedback=f"Wrong accuracy for the 1st model")

        # 2nd model
        model_name_answer = 'LogisticRegression'
        if model_name_answer not in relevant_lines[2]:
            model_name_reply = get_model_name(relevant_lines[2])
            return CheckResult.wrong(feedback=f"Incorrect name of the 2nd model\n"
                                              f"Expected {model_name_answer}, found {model_name_reply}")
        

        accuracy_reply = re.findall(r'\d*\.\d+|\d+', relevant_lines[3])
        if len(accuracy_reply) != 1:
            return CheckResult.wrong(feedback=f'It should be one number in the "Accuracy:" section')
        # 1% error rate is allowed, right accuracy = 0.918
        if not 0.99 * 0.918 < float(accuracy_reply[0]) < 1.01 * 0.918:
            return CheckResult.wrong(feedback=f"Wrong accuracy for the 2nd model")

        return CheckResult.correct()


if __name__ == '__main__':
    FakeNewsTest().run_tests()