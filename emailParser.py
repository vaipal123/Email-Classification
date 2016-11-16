import email
import os

for filename in os.listdir('ham'):
    print filename   
    with open('ham/' + filename , 'r') as A:
        data=A.read()
    # print data
    # b = email.message_from_string(data)
    # if b.is_multipart():
    #     for payload in b.get_payload():
    #         # if payload.is_multipart(): ...
    #         print payload.get_payload()
    # else:
    #     print b.get_payload()

    b = email.message_from_string(data)
    body = ""

    if b.is_multipart():
        for part in b.walk():
            ctype = part.get_content_type()
            cdispo = str(part.get('Content-Disposition'))

            # skip any text/plain (txt) attachments
            if ctype == 'text/plain' and 'attachment' not in cdispo:
                body = part.get_payload(decode=True)  # decode
                break
    # not multipart - i.e. plain text, no attachments, keeping fingers crossed
    else:
        body = b.get_payload(decode=True)


    with open('ham/' + filename, 'r+') as f:
        f.seek(0)
        f.write(body)
        f.truncate()
    # with open('a.txt', 'r') as A:
    #     data=A.read()
    # print data        

