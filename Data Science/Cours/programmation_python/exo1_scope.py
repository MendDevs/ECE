def my_function(a):
    b = a - 2
    return b

c = 3

if c > 2:
    d = my_function(5)
    print(d)
    
#a is local, vie tempraire (pendant l'appel), n'exsite pas si c = 1 (becuase the function won't be called).
# same as a
#c is global, tt la durée du script 
#d global, tt la durée du script. If c = 1, d won't be created