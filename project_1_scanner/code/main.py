import scanner
from scanner import DocumentFormat

documentscanner = scanner.DocumentScanner(
    "scanner/math.JPEG",
    DocumentFormat.A4,
    True)
img = documentscanner.scan()
documentscanner.show_image_history()
