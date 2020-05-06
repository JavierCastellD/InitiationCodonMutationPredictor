function realizarPrediccion(){
    // Obtenemos las distintas variables del formulario
    var lmiu = document.getElementById("lostMetsInUTR").value
    var psc = document.getElementById("prematureStopCodon").value
    var msl = document.getElementById("mutatedSequenceLength").value
    var rfs = document.getElementById("readingFrameStatus").value
    
    var resultado = document.getElementById("resultado")
    var error = document.getElementById("error")

    // Comprobamos si los valores son correctos
    if (!isNaN(lmiu) && Number(lmiu) >= 0) {
        if (!isNaN(msl) && Number(msl) >= 0) {

            // En este caso los valores son correctos, por lo que hacemos 
            // una petición al endpoint para que se ejecute el script con los datos
            error.style.visibility = "hidden"
            
            const Http = new XMLHttpRequest()
            const url = 'http://localhost:5000/'
            const variables = '?lmiu='+lmiu+'&psc='+psc+'&msl='+msl+'&rfs='+rfs

            Http.open("GET", url+variables)
            Http.send()

            Http.onreadystatechange = (e) => {
                if (Http.readyState == 4 && Http.status == 200){
                    resultado.textContent = "La mutación es " + Http.responseText
                    // "La mutación es DELETEREA"
                    // "La mutación es BENIGNA"
                }
            }
            
        } else {
            // En este caso, hay un problema con el valor de MSL
            error.style.visibility = "visible"
            error.textContent = "Mutated Sequence Length tiene que ser mayor que 0."
        }

    } else {
        // En este caso, hay un problema con el valor de LMIU
        error.style.visibility = "visible"
        error.textContent = "Lost Mets in UTR tiene que ser un entero mayor o igual a 0."
    }

}