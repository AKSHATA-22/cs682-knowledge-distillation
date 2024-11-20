Milestone:

Your project milestone report should be between 2 - 3 pages using the template (pdf, latex source). The following is a suggested structure for your report:
Title, Author(s)

Introduction: This section introduces your problem, motivation, and the overall plan. It should describe your problem precisely specifying the dataset to be used, expected results and evaluation.

## Copy from Proposal



Related work: A literature survey of past work on this topic. Introduce the baselines you will compare to and the weakness you plan to address. This section should be nearly complete.

## Copy from Proposal


Technical Approach: Describe the methods you intend to apply to solve the given problem.

## Two methods of doing KD:
    - Mutual Information
    - Using KL divergence and Softmax Loss
## Two architectures using the above two methods:
    - Teacher - TA - Student
    - TN1 (uses MI) + TN2 (uses KLD) - Student

Intermediate/Preliminary Results: State and evaluate your results upto the milestone.
Submission: Please upload a PDF file to Gradescope. Please coordinate with your teammates and submit only under ONE of your accounts, and add your teammates on Gradescope.

 # Outputs:

 Teacher VGG: 
 1. Best Validation Accuracy: 76.80
 2. Test Accuracy: 59.45

 Teaching Assistant: 
 1. Best Validation Accuracy: 84.00
 2. Test Accuracy: 70.12