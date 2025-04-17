print("1. Soal Dasar : Input & Output")
nama = input("Nama mu : ")
umur = input("Umur : ")
tinggi = input("Tinggi Badan : ")

try:
    umur = float(umur)
    tinggi = float(tinggi)
except:
    print("Please check your input for your age and height, make sure they are both a number")

print(f"Nama mu adalah {nama}, berumur {umur} dan memiliki tinggi {tinggi}")

print("\nNow Enter Values to calculate an area of rectangle")
width = input("Enter for width : ")
length = input("Enter for length : ") 

try:
    width = float(width)
    length = float(length)
except:
    print("Please check your input for the width and length, make sure they are both a number")


print(f"\nArea of rectangle {width * length}\n")

print("2. Soal Kondisional (if - else)")
if umur < 10:
    print("Kamu adalah anak kecil")
elif umur >= 11 and umur <= 17:
    print("Kamu adalah remaja")
elif umur > 17 and umur <= 50:
    print("Kamu adalah dewasa")
elif umur > 50:
    print("Kamu adalah Lansia")
else:
    print("Invalid Age")

print("\n3. Soal Perulangan (Loop)")

print("\nA. For Loop")

for i in range(11):
    print(i)

print("\nB. For Loop Break")
for i in range(7):
    print(i)
    if i == 5:
        break

print("\nC. While Loop")
i = 0
while i < 10:
    if i == 5:
        break
    print(i)
    i += 1

print("4. Soal Fungsi (def)")
def multiplication(a, b):
    return a * b
print(f"Multiplication of 5 and 4 is {multiplication(4,5)}")
