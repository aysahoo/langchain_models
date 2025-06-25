from pydantic import BaseModel
from typing import Optional

class Student(BaseModel):

    name : str

new_student = {'name' : 'nitish'}

student = Student(**new_student)

print(student)

print(student.name)