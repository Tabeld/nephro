from Mark import Mark

from Line import Line
from Vector import Vector


class Area:
    def __init__(self):
        self.vertices = list()
        self.edges = list()
        self.mark = Mark()
        self.vectors_to_vertex = dict()

    def add_points(self, points):
        if len(points) == 0:
            return
        points.pop(-1)
        self.vertices.append(Vector(points[0][0], points[0][1]))
        clockwise = self.calculate_polygon_area(points) < 0
        for i in range(1, len(points)):
            point = points[i]
            last_point = points[i - 1]
            vertex1 = Vector(point[0], point[1])
            vertex2 = Vector(last_point[0], last_point[1])
            self.vertices.append(vertex1)
            self.edges.append(Line(vertex2, vertex1, clockwise))
        self.edges.append(Line(self.vertices[-1], self.vertices[0], clockwise))
        for i in range(1, len(self.edges)):
            vector = (self.edges[i].normal + self.edges[i - 1].normal) * 0.25
            self.vectors_to_vertex[self.edges[i].start] = vector
        self.vectors_to_vertex[self.edges[-1].end] = (self.edges[0].normal + self.edges[
            -1].normal) * 0.25

    def set_mark(self, mark):
        self.mark = mark

    def pointInsideShape(self, targetPoint):
        '''Проверить, что точка лежит внутри фигуры, образуемой списком точек'''
        minDistance = float("inf")
        targetLine = None
        projectPoint = None
        for line in self.edges:
            h = line.getProjectionFrom(targetPoint)
            distance = targetPoint.calcDistance(h)
            if distance < minDistance:
                minDistance = distance
                targetLine = line
                projectPoint = h

        if projectPoint == targetLine.start:
            normal = self.vectors_to_vertex[targetLine.start]
        elif projectPoint == targetLine.end:
            normal = self.vectors_to_vertex[targetLine.end]
        else:
            normal = targetLine.normal
        directLine = projectPoint - targetPoint
        scalarRes = normal.calcScalarProduct(directLine)
        return scalarRes > 0 and targetLine.pointOnSegment(projectPoint)

    def calculate_polygon_area(self, vertices):
        n = len(vertices)
        area = 0.0
        for i in range(n):
            x1, y1 = vertices[i][0], vertices[i][1]
            x2, y2 = vertices[(i + 1) % n][0], vertices[(i + 1) % n][1]
            area += (x1 * y1) - (x2 * y2)
        return area
