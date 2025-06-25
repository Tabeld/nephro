import math
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f"{{x: {self.x}; y: {self.y}}}"

    def __eq__(self, other) -> bool:
        return self.x == other.x and self.y == other.y

    def __ne__(self, other) -> bool:
        return self.x != other.x or self.y != other.y

    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other):
        return Vector(self.x * other, self.y * other)

    def mulOnConst(self, k: int):
        return Vector(self.x * k, self.y * k)

    def calcScalarProduct(self, other) -> float:
        return self.x * other.x + self.y * other.y

    def calcDistance(self, target) -> float:
        return (target - self).getLength()

    def getLength(self) -> float:
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def normalize(self):
        length = self.getLength()
        return Vector(self.x / length, self.y / length)

    def __hash__(self):
        return hash(tuple([self.x, self.y]))