# medical_chatbot

Medical abstracts describe the current conditions of a patient. Doctors routinely scan dozens or
hundreds of abstracts each day as they do their rounds in a hospital and must quickly pick up on
the salient information pointing to the patient’s malady. You are trying to design assistive
technology that can identify, with high precision, the class of problems described in the
abstract.<br />
In the given dataset, abstracts from 5 different conditions (consider these as classes) have been
included:<br />
1- digestive system diseases<br />
2- cardiovascular diseases<br />
3- neoplasms<br />
4- nervous system diseases<br />
5- general pathological conditions<br />
Data files are hosted at - https://github.com/EvolentGenAIteam/GenAIData<br />
Training data has 2 columns first is class/condition and second is medical text.<br />
Testing data will have free clinical text for which your chat-bot should be able to determine which
condition the text belongs to.<br />
(assume each paragraph is clinical text from different patient)<br />
LLMs Pipeline Design:<br />
A. Use train data to create conversational chat-bot platform with the help of any LLM
that can classify conditions of the patients. (Your LLM should consider this training
data before answering the questions)<br />
B. Use test data to predict what should be ‘classification’ categories of the condition<br />
C. Create small UI to showcase the response.<br />
