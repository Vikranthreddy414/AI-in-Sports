system_prompt = """<|INST|><|SYS|>
You are a helpful, respectful, and honest assistant for labeling topics in a diverse range of emails from Paycom. Your goal is to identify and summarize the main topics discussed in the emails accurately.
<|/SYS|>
"""

example_prompt_1 = """<|INST|>
I have a topic that contains the following documents:
- "Please find attached your monthly payroll report."
- "Reminder: The deadline for payroll submission is approaching."
- "Your payroll for this month has been processed successfully."
- "We have updated the payroll schedule for the next quarter. Please review."

The topic is described by the following keywords: 'payroll, report, submission, processed, schedule, monthly, deadline'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Payroll Processing and Schedule
"""

example_prompt_2 = """<|INST|>
I have a topic that contains the following documents:
- "The benefits enrollment period is now open."
- "Please review your benefits plan and make any necessary changes."
- "Your health insurance details for the upcoming year are attached."
- "Thank you for enrolling in the employee benefits program."

The topic is described by the following keywords: 'benefits, enrollment, health insurance, plan, employee, program'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Employee Benefits and Enrollment


example_prompt_3 = """<|INST|>
I have a topic that contains the following documents:
- "Enroll in our new leadership training program."
- "The webinar on effective communication in the workplace is now available for replay."
- "Join our upcoming workshop on time management skills."
- "Please complete the mandatory cybersecurity training by the end of the month."

The topic is described by the following keywords: 'training, development, webinar, workshop, leadership, communication, cybersecurity'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Training and Development Programs

example_prompt_4 = """<|INST|>
I have a topic that contains the following documents:
- "Thank you for choosing Paycom. We are here to assist you with any payroll needs."
- "Your account manager will be in touch shortly to discuss the next steps."
- "Please provide feedback on our service through the attached survey."
- "Here is your monthly usage report."

The topic is described by the following keywords: 'client, communication, feedback, service, account manager, usage report'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Client Communication and Feedback

example_prompt_5 = """<|INST|>
I have a topic that contains the following documents:
- "You are invited to the quarterly business review meeting."
- "Please confirm your availability for the team meeting next week."
- "The meeting has been rescheduled to 3 PM on Friday."
- "Here is the agenda for the upcoming project kickoff meeting."

The topic is described by the following keywords: 'meeting, invitation, schedule, availability, agenda, business review'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST] Meeting Invitations and Scheduling
