function realizarPrediccionPorCaracteristicas(){
    // Obtenemos las distintas variables del formulario
    var lmiu = document.getElementById("lostMetsInUTR").value
    var psc = document.getElementById("prematureStopCodon").value
    var msl = document.getElementById("mutatedSequenceLength").value
    var rfs = document.getElementById("readingFrameStatus").value
    var mp = document.getElementById("metPosition").value
    var scp = document.getElementById("stopCodonPosition").value
    
    var resultado = document.getElementById("resultadoPorCaracteristicas")
    var error = document.getElementById("errorPorCaracteristicas")

    // Comprobamos si los valores son correctos
    if (!isNaN(lmiu) && Number(lmiu) >= 0) {
        if (!isNaN(msl) && Number(msl) >= 0) {
            if (!isNaN(mp) && Number(mp) >= 0){
                if (!isNaN(scp) && Number(scp) >= 0){
                    // En este caso los valores son correctos, por lo que hacemos 
                    // una petición al endpoint para que se ejecute el script con los datos
                    error.style.visibility = "hidden"
                    
                    const Http = new XMLHttpRequest()
                    const url = 'http://localhost:5000/prediccionPorCaracteristicas'
                    const variables = '?lmiu='+lmiu+'&psc='+psc+'&msl='+msl+'&rfs='+rfs+'&mp='+mp+'&scp='+scp

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
                    // En este caso, hay un problema con el valor de SCP
                    error.style.visibility = "visible"
                    error.textContent = "Stop Codon Position tiene que ser mayor que 0."
                }
                
            } else {
                // En este caso, hay un problema con el valor de MP
                error.style.visibility = "visible"
                error.textContent = "Met Position tiene que ser mayor que 0."
                
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


function realizarPrediccionPorSecuencias(){
    // Obtenemos las distintas variables del formulario
    var cdna = document.getElementById("cdna").value
    var cds = document.getElementById("cds").value
    var mutatedCdna = document.getElementById("mutatedCdna").value
    var resultado = document.getElementById("resultadoPorSecuencias")
    var error = document.getElementById("errorPorSecuencias")

    const xhttp = new XMLHttpRequest()
    const url = 'http://localhost:5000/prediccionPorSecuencias'
    const variables = 'cdna='+cdna+'&cds='+cds+'&mutatedCdna='+mutatedCdna

    xhttp.open("POST", url)
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send(variables);

    xhttp.onreadystatechange = (e) => {
        if (Http.readyState == 4 && Http.status == 200){
            resultado.textContent = "La mutación es " + Http.responseText
            // "La mutación es DELETEREA"
            // "La mutación es BENIGNA"
        }
    }
}

function realizarPrediccionPorEnsemblID(){
    var transcriptID = document.getElementById("transcriptID").value
    var cambioCodon = document.getElementById("cambioCodon").value
    var error = document.getElementById("errorPorEnsemblID")
    var resultado = document.getElementById("resultadoPorEnsemblID")

    const Http = new XMLHttpRequest()
    const url = 'http://localhost:5000/prediccionPorSeqID'
    const variables = '?transcriptId='+transcriptID+'&cambioCodon='+cambioCodon

    Http.open("GET", url+variables)
    Http.send()

    Http.onreadystatechange = (e) => {
        if (Http.readyState == 4 && Http.status == 200){
            resultado.textContent = "La mutación es " + Http.responseText
            // "La mutación es DELETEREA"
            // "La mutación es BENIGNA"
        }
    }
}