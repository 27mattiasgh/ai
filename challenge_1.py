things = [8, 7, 21, 4, -2, 0, 40]

for i in range(len(things) - 1):
    for x in range(0, len(things) - 1):  
        if things[x] > things[x + 1]:
            temp = things[x]
            things[x] = things[x + 1] 
            things[x + 1] = temp

print(things)
