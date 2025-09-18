total = 3

a_number = 5              # a_number becomes 5
a_number = total          # a_number becomes the value of total
a_number = total + 5      # a_number becomes the value of total + 5
a_number = a_number + 1   # a_number becomes the value of a_number + 1


# these are all illegal:
3 = 4
3 = a
a + b = 3

# both a and b will be set to zero:
a = b = 0

# this is illegal, because we can't set 0 to b:
a = 0 = b