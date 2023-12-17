import init, { Model } from "./build/m.js";

function fixTwo(x) { return Math.floor(x * 100) / 100 } 

function humanSize(size) {
    if(size < 1e3) return `${fixTwo(size)}b`
    if(size < 1e6) return `${fixTwo(size/1e3)}kb`
    if(size < 1e9) return `${fixTwo(size/1e6)}mb`
    if(size < 1e12) return `${fixTwo(size/1e9)}gb`
    return `${fixTwo(size/1e12)}tb`
}

function humanTime(seconds) {
    const _year = 31536e3
    const _mon = 2592e3
    const _day = 864e2
    const _hour = 36e2
    const _min = 60
    const _sec = 1

    const year_rem = seconds % _year
    const years = (seconds - year_rem) / _year

    const month_rem = year_rem % _mon
    const months = (year_rem - month_rem) / _mon

    const day_rem = month_rem % _day
    const days = (month_rem - day_rem) / _day

    const hour_rem = day_rem % _hour
    const hours = (day_rem - hour_rem) / _hour

    const minute_rem = hour_rem % _min
    const minutes = (hour_rem - minute_rem) / _min

    const second_rem = minute_rem % _sec
    const second = (minute_rem - second_rem) / _sec

    return (years > 0 ? `${years} year${years == 1 ? '' : 's'} ` : '') + (months > 0 ? `${months} month${months == 1 ? '' : 's'} `: '') +
        (days > 0 ? `${days} day${days == 1 ? '' : 's'} ` : '') + (hours > 0 ? `${hours} hour${hours == 1 ? '' : 's'} ` : '') +
        (minutes > 0 ? `${minutes} minute${minutes == 1 ? '' : 's'} ` : '') + (seconds > 0 ? `${second} second${second == 1 ? '' : 's'} ` : '')
}

let lastSend = 0
let lastTime = Infinity
let times = [0, 0, 0, 0]

async function fetchArrayBuffer(url) {
    const cacheName = "phi-mixformer-candle-cache";
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(url);
    if (cachedResponse) {
        const data = await cachedResponse.arrayBuffer();
        return new Uint8Array(data);
    }
    const res = await fetch(url, { cache: "force-cache" });
    while (!res.body) { }
    const reader = res.body.getReader();
    const contentLength = +(res.headers.get('Content-Length') ?? 0);
    let receivedLength = 0;
    let chunks = [];
    while (true) {
        const { done, value } = await reader.read();
        if (done) {
            break;
        }
        chunks.push(value);
        receivedLength += value.length;
        if(Date.now() - lastSend > 250) {
            times.push(receivedLength)
            times = times.slice(1)
            let max = [times[3] - times[2], times[2] - times[1], times[1] - times[0]]
            let median = (max[0] + max[1] + max[2]) / 3
            let lengthPerSecond = median * 4
            let leftSize = contentLength - receivedLength
            let leftTime = Math.abs(leftSize / lengthPerSecond)

            if(leftTime > lastTime * 1.5 && lastTime != 0) leftTime = lastTime * 1.2
            // if(leftTime > lastTime) leftTime = lastTime
            lastTime = leftTime
            let downloadMessage = `Downloading... ${fixTwo((receivedLength / contentLength) * 100)}% (${humanSize(Math.floor(receivedLength * 100) / 100)})
Estimated time remaining: ${humanTime(leftTime)} (may be inaccurate)
Total size: ${humanSize(fixTwo(contentLength))}
Download URL: ${url}`
            self.postMessage({ status: "loading", message: downloadMessage })
            // console.log(downloadMessage)
            lastSend = Date.now()
        }
    }
    let chunksAll = new Uint8Array(receivedLength);
    let position = 0;
    for (let chunk of chunks) {
        chunksAll.set(chunk, position);
        position += chunk.length;
    }
    cache.put(url, new Response(chunksAll));
    return chunksAll;
}

async function concatenateArrayBuffers(urls) {
    const arrayBuffers = await Promise.all(urls.map(url => fetchArrayBuffer(url)));

    let totalLength = arrayBuffers.reduce((acc, arrayBuffer) => acc + arrayBuffer.byteLength, 0);
    let concatenatedBuffer = new Uint8Array(totalLength);

    let offset = 0;
    arrayBuffers.forEach(buffer => {
        concatenatedBuffer.set(new Uint8Array(buffer), offset);
        offset += buffer.byteLength;
    });
    return concatenatedBuffer;
}

class Phi {
    static instance = {};

    static async getInstance(
        weightsURL,
        modelID,
        tokenizerURL,
        configURL,
        quantized
    ) {
        // load individual modelID only once
        if (!this.instance[modelID]) {
            await init();

            self.postMessage({ status: "loading", message: "Loading Model" });
            const [weightsArrayU8, tokenizerArrayU8, configArrayU8] =
                await Promise.all([
                    weightsURL instanceof Array ? concatenateArrayBuffers(weightsURL) : fetchArrayBuffer(weightsURL),
                    fetchArrayBuffer(tokenizerURL),
                    fetchArrayBuffer(configURL),
                ]);

            this.instance[modelID] = new Model(
                weightsArrayU8,
                tokenizerArrayU8,
                configArrayU8,
                quantized
            );
        }
        return this.instance[modelID];
    }
}

let controller = null;
self.addEventListener("message", (event) => {
    if (event.data.command === "start") {
        controller = new AbortController();
        generate(event.data);
    } else if (event.data.command === "abort") {
        controller.abort();
    }
});

async function generate(data) {
    const {
        weightsURL,
        modelID,
        tokenizerURL,
        configURL,
        quantized,
        prompt,
        temp,
        top_p,
        repeatPenalty,
        seed,
        maxSeqLen,
    } = data;
    try {
        self.postMessage({ status: "loading", message: "Starting Phi" });
        const model = await Phi.getInstance(
            weightsURL,
            modelID,
            tokenizerURL,
            configURL,
            quantized
        );

        self.postMessage({ status: "loading", message: "Initializing model" });
        const firstToken = model.init_with_prompt(
            prompt,
            temp,
            top_p,
            repeatPenalty,
            64,
            BigInt(seed)
        );
        const seq_len = 2048;

        let sentence = firstToken;
        let maxTokens = maxSeqLen ? maxSeqLen : seq_len - prompt.length - 1;
        let startTime = performance.now();
        let tokensCount = 0;
        while (tokensCount < maxTokens) {
            await new Promise(async (resolve) => {
                if (controller && controller.signal.aborted) {
                    self.postMessage({
                        status: "aborted",
                        message: "Aborted",
                        output: prompt + sentence,
                    });
                    return;
                }
                const token = await model.next_token();
                if (token === "<|endoftext|>") {
                    self.postMessage({
                        status: "complete",
                        message: "complete",
                        output: prompt + sentence,
                    });
                    return;
                }
                const tokensSec =
                    ((tokensCount + 1) / (performance.now() - startTime)) * 1000;

                sentence += token;
                self.postMessage({
                    status: "generating",
                    message: "Generating token",
                    token: token,
                    sentence: sentence,
                    totalTime: performance.now() - startTime,
                    tokensSec,
                    prompt: prompt,
                });
                setTimeout(resolve, 0);
            });
            tokensCount++;
        }
        self.postMessage({
            status: "complete",
            message: "complete",
            output: prompt + sentence,
        });
    } catch (e) {
        self.postMessage({ error: e });
    }
}
