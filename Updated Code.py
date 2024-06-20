CREATE CONSTRAINT ON (p:Person) ASSERT p.email_address IS UNIQUE;
CREATE CONSTRAINT ON (e:Email) ASSERT e.emailid IS UNIQUE;
CREATE CONSTRAINT ON (l:Label) ASSERT l.name IS UNIQUE;

// Load and create Email nodes
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
MERGE (e:Email {emailid: row.emailid})
ON CREATE SET e.fromaddress = row.fromaddress,
              e.to = row.to,
              e.subject = row.subject,
              e.body_content = row.body_content,
              e.sentdate = datetime(row.sentdate),
              e.label_scores = row.label_scores;

// Create Person nodes for 'fromaddress' and 'to'
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
MERGE (sender:Person {email_address: row.fromaddress})
MERGE (recipient:Person {email_address: row.to})
WITH row, sender, recipient
MATCH (e:Email {emailid: row.emailid})
MERGE (e)-[:SENT_BY]->(sender)
MERGE (e)-[:SENT_TO]->(recipient);

// Create Person nodes for 'cc'
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
WITH row, apoc.convert.fromJsonList(row.cc) AS ccAddresses
UNWIND ccAddresses AS ccAddress
MERGE (ccPerson:Person {email_address: ccAddress})
WITH row, ccPerson
MATCH (e:Email {emailid: row.emailid})
MERGE (e)-[:CC]->(ccPerson);

// Create Person nodes for 'bcc'
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
WITH row, apoc.convert.fromJsonList(row.bcc) AS bccAddresses
UNWIND bccAddresses AS bccAddress
MERGE (bccPerson:Person {email_address: bccAddress})
WITH row, bccPerson
MATCH (e:Email {emailid: row.emailid})
MERGE (e)-[:BCC]->(bccPerson);

// Create REPLY_TO relationships
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
WITH row, apoc.convert.fromJsonMap(row.inreplyto) AS inreplyToEmail
MATCH (e:Email {emailid: row.emailid})
MATCH (replyTo:Email {emailid: apoc.text.replace(inreplyToEmail, '[<>]', '')})
MERGE (e)-[:REPLY_TO]->(replyTo);

// Create REFERENCES relationships
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
WITH row, apoc.convert.fromJsonList(row.references) AS referenceEmails
UNWIND referenceEmails AS refEmail
MATCH (e:Email {emailid: row.emailid})
MATCH (ref:Email {emailid: apoc.text.replace(refEmail, '[<>]', '')})
MERGE (e)-[:REFERENCES]->(ref);

// Create Label nodes and HAS_LABEL relationships
LOAD CSV WITH HEADERS FROM 'file:///large_emails_dataset.csv' AS row
WITH row, apoc.convert.fromJsonMap(row.label_scores) AS labelKeyValue
UNWIND keys(labelKeyValue) AS labelName
MERGE (label:Label {name: labelName})
WITH row, label, labelKeyValue[labelName] AS score
MATCH (e:Email {emailid: row.emailid})
MERGE (e)-[r:HAS_LABEL]->(label)
SET r.score = score;




import csv
import json
from datetime import datetime
from neomodel import db, StructuredNode, StringProperty, JSONProperty, DateTimeProperty, UniqueIdProperty, RelationshipTo, StructuredRel, config
from neomodel.exceptions import UniqueProperty, DoesNotExist
import logging

# Set up the connection to the database
config.DATABASE_URL = 'bolt://neo4j:password@localhost:7687'

# Enable logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Person(StructuredNode):
    email_address = StringProperty(unique_index=True, required=True)

class Label(StructuredNode):
    name = StringProperty(unique_index=True, required=True)

class LabelRel(StructuredRel):
    score = StringProperty()

class Email(StructuredNode):
    emailid = UniqueIdProperty()
    fromaddress = StringProperty(required=True)
    to = StringProperty(required=True)
    subject = StringProperty()
    body_content = StringProperty()
    sentdate = DateTimeProperty()
    label_scores = JSONProperty()

    sent_by = RelationshipTo('Person', 'SENT_BY')
    sent_to = RelationshipTo('Person', 'SENT_TO')
    cc = RelationshipTo('Person', 'CC')
    bcc = RelationshipTo('Person', 'BCC')
    reply_to = RelationshipTo('Email', 'REPLY_TO')
    references = RelationshipTo('Email', 'REFERENCES')
    has_label = RelationshipTo('Label', 'HAS_LABEL', model=LabelRel)

def load_data(csv_file_path):
    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for row in reader:
                try:
                    # Create Email node
                    email_node = Email(
                        emailid=row['emailid'],
                        fromaddress=row['fromaddress'],
                        to=row['to'],
                        subject=row['subject'],
                        body_content=row['body_content'],
                        sentdate=datetime.strptime(row['sentdate'], '%Y-%m-%d %H:%M:%S'),
                        label_scores=json.loads(row['label_scores'])
                    ).save()

                    # Create or get Person nodes
                    sender_node, _ = Person.get_or_create({'email_address': row['fromaddress']})
                    recipient_node, _ = Person.get_or_create({'email_address': row['to']})
                    email_node.sent_by.connect(sender_node)
                    email_node.sent_to.connect(recipient_node)

                    # CC addresses
                    for cc_address in json.loads(row['cc']):
                        cc_node, _ = Person.get_or_create({'email_address': cc_address})
                        email_node.cc.connect(cc_node)

                    # BCC addresses
                    for bcc_address in json.loads(row['bcc']):
                        bcc_node, _ = Person.get_or_create({'email_address': bcc_address})
                        email_node.bcc.connect(bcc_node)

                    # Reply to
                    if row['inreplyto']:
                        inreplyto_emailid = row['inreplyto'].strip('<>')
                        try:
                            inreplyto_email = Email.nodes.get(emailid=inreplyto_emailid)
                            email_node.reply_to.connect(inreplyto_email)
                        except DoesNotExist:
                            logger.warning(f"Referenced email {inreplyto_emailid} does not exist")

                    # References
                    for ref in json.loads(row['references']):
                        ref_emailid = ref.strip('<>')
                        try:
                            ref_email = Email.nodes.get(emailid=ref_emailid)
                            email_node.references.connect(ref_email)
                        except DoesNotExist:
                            logger.warning(f"Referenced email {ref_emailid} does not exist")

                    # Labels
                    for label, score in json.loads(row['label_scores']).items():
                        label_node, _ = Label.get_or_create({'name': label})
                        rel = email_node.has_label.connect(label_node)
                        rel.score = score
                        rel.save()

                except UniqueProperty:
                    logger.warning(f"Email with ID {row['emailid']} already exists.")
                except Exception as e:
                    logger.error(f"Error processing row {row}: {e}")

    except FileNotFoundError:
        logger.error(f"File {csv_file_path} not found.")
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")

# Load the data from the CSV file
if __name__ == "__main__":
    load_data('/path/to/large_emails_dataset.csv')


Nodes
Email:

emailid: Unique identifier for the email.
fromaddress: Email address of the sender.
to: Email address of the recipient.
subject: Subject of the email.
body_content: Content of the email.
sentdate: Date and time the email was sent.
label_scores: JSON string of labels and their scores.
Person:

email_address: Unique email address of the person.
Label:

name: Name of the label (e.g., "anger", "disgust").
score: Score associated with the label for a particular email.
Relationships
SENT_BY:

From: Email
To: Person
SENT_TO:

From: Email
To: Person
CC:

From: Email
To: Person
BCC:

From: Email
To: Person
REPLY_TO:

From: Email
To: Email
REFERENCES:

From: Email
To: Email
HAS_LABEL:

From: Email
To: Label