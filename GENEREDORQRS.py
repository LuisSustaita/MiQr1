import qrcode

# ***** Crear QR
borde = 6
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_M,
    box_size=borde,
    border=borde,
)
qr.add_data('BEGIN:VCARD \nN:Luis;Sustaita \nTEL;CELL:4491046191 \nORG:UG;Sistemas de informacion \nTITLE:Lic. \nEMAIL:ld.delacruzsustaita@ugto.com \nBDAY:19960611 \nEND:VCARD')

qr.make(fit=True)
img = qr.make_image()
rutaQRs="C:/Users/luisd/Desktop/tesis/tesis latex/imagenes/qrs/"
NOMBRE_DE_QR = "card.png"
img.save(rutaQRs+NOMBRE_DE_QR)