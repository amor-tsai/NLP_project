from QuestionGenerator import QuestionGenerator
import argparse


def run(inputSent=None,inputFilename=None,outputFilename=None):
    qg = QuestionGenerator()
    questions = []
    if inputSent:
        questions = qg.getQuestions(inputSent)
    elif inputFilename:
        inputs = open(inputFilename, 'r').readlines()
        for inputSent in inputs:
            tmp = qg.getQuestions(inputSent.strip())
            for q in tmp:
                questions.append(q)
            questions.append('-------------------------------------')
    if outputFilename:
        with open(outputFilename,'w') as f:
            for q in questions:
                f.writelines(q)
                f.write('\n')
    else:
        for q in questions:
            print(q)


def parseArgs():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Question Generator')
    parser.add_argument(
        '-sentence', default=None, help='Single sentence input')
    parser.add_argument(
        '-input_file', default=None, help='Sentences file input')
    parser.add_argument(
        '-output', default=None, help='Output file name')
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseArgs()
    if args.sentence:
        print('Single sentence input')
        run(inputSent=args.sentence,outputFilename=args.output)
    elif args.input_file:
        print('sentences file input')
        run(inputFilename=args.input_file, outputFilename=args.output)
    else:
        raise Exception(
            ('You must provide one of the two options: -sentence, -input_file'))