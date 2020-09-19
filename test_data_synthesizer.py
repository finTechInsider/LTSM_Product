import random
import uuid 
  
outF = open("input_data.txt", "w")
for i in range(5000):
    id_type = random.choice(['BBG', 'ISIN', 'UUID', 'DTCCC'])
    id = None

    if id_type == 'BBG':
        id = 'BBG{}'.format('{:05}'.format(abs(random.randint(0, 10000))))
    elif id_type == 'ISIN':
        id = 'US{}'.format('{:08}'.format(abs(random.randint(0, 10000))))
    elif id_type == 'DTCCC':
        id = '{}'.format('{:10}'.format(abs(random.randint(0, 10000000000))))
    elif id_type == 'UUID':
        id = str(uuid.uuid1())

    outF.write("{},{}\n".format(id, id_type))

outF.close()