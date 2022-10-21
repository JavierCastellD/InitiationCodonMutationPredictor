function realizarPrediccionPorCaracteristicas(){
    // Obtain the different variables from the form
    var psc = document.getElementById("prematureStopCodon").value
    var rfs = document.getElementById("readingFrameStatus").value
    var lm5 = document.getElementById("nLostMets5UTR").value
    var msl = document.getElementById("mutatedSequenceLength").value
    var mp = document.getElementById("metPosition").value
    var scp = document.getElementById("stopCodonPosition").value
    
    var resultado = document.getElementById("resultadoPorCaracteristicas")
    var error = document.getElementById("errorPorCaracteristicas")

    // Check if the values are correct
    if (!isNaN(lm5) && Number(lm5) >= 0) {
        if (!isNaN(msl) && Number(msl) >= 0) {
            if (!isNaN(mp) && Number(mp) >= 0){
                if (!isNaN(scp) && Number(scp) >= 0){
                    // If so, we do a petition to the endpoint so that it
                    // executes the script with this data
                    error.style.visibility = "hidden"
                    
                    const Http = new XMLHttpRequest()
                    // TODO: Cambiar localhost por la IP que diga
                    const url = 'http://semantics.inf.um.es:5000/prediccionPorCaracteristicas'
                    const variables = '?lm5='+lm5+'&msl='+msl+'&mp='+mp+'&scp='+scp+'&psc='+psc+'&rfs='+rfs

                    Http.open("GET", url+variables)
                    Http.send()
                        
                    Http.onreadystatechange = (e) => {
                        if (Http.readyState == 4 && Http.status == 200){
                            resultado.textContent = "The mutation is " + Http.responseText
                        }
                    }
                } else {
                    // There is a problem with Stop Codon Positions
                    error.style.visibility = "visible"
                    error.textContent = "Stop Codon Position has to be an integer greater than 0."
                }
                
            } else {
                // There is a problem with Met Position
                error.style.visibility = "visible"
                error.textContent = "Met Position has to be an integer greater than 0."
                
            }
            
        } else {
            // There is a problem with Mutated Sequence Length
            error.style.visibility = "visible"
            error.textContent = "Mutated Sequence Length has to be greater or equal than 0."
        }

    } else {
        // There is a problem with NMETS_5_UTR
        error.style.visibility = "visible"
        error.textContent = "Number of Mets in 5' UTR has to be an integer equal or greater than 0."
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
    const url = 'http://semantics.inf.um.es:5000/prediccionPorSecuencias'
    const variables = 'cdna='+cdna+'&cds='+cds+'&mutatedCdna='+mutatedCdna

    xhttp.open("POST", url)
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send(variables);

    xhttp.onreadystatechange = (e) => {
        if (Http.readyState == 4 && Http.status == 200){
            resultado.textContent = "The mutation is " + Http.responseText
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
    const url = 'http://semantics.inf.um.es:5000/prediccionPorSeqID'
    const variables = '?transcriptId='+transcriptID+'&cambioCodon='+cambioCodon

    Http.open("GET", url+variables)
    Http.send()

    Http.onreadystatechange = (e) => {
        if (Http.readyState == 4 && Http.status == 200){
            resultado.textContent = "The mutation is " + Http.responseText
            // "La mutación es DELETEREA"
            // "La mutación es BENIGNA"
        }
    }
}
