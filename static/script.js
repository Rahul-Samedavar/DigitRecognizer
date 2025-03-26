document.addEventListener('DOMContentLoaded', function () {
    const canvas = document.getElementById('drawingCanvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clearButton');
    const resultsBody = document.getElementById('resultsBody');

    const canvasSize = { width: 280, height: 280 };
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;

    const gridSize = 28;

    let isDrawing = false;

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.lineWidth = 18;
    ctx.strokeStyle = 'black';

    function clearCanvas() {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvasSize.width, canvasSize.height);
        resultsBody.innerHTML = `
            <tr class="empty-state">
                <td colspan="2">Draw something to see classification results</td>
            </tr>
        `;
    }

    clearCanvas();

    function getBoundingBox(imageData) {
        const { data, width, height } = imageData;
        let minX = width, minY = height, maxX = 0, maxY = 0;
        let found = false;

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                const alpha = data[idx + 3];  
                const isBlack = data[idx] < 128 && alpha > 0;

                if (isBlack) {
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;
                    found = true;
                }
            }
        }

        if (!found) return null;

        return {
            x: minX,
            y: minY,
            width: maxX - minX,
            height: maxY - minY,
        };
    }

    function canvasToArray() {
        const imageData = ctx.getImageData(0, 0, canvasSize.width, canvasSize.height);
        const box = getBoundingBox(imageData);

        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        tempCanvas.width = canvasSize.width;
        tempCanvas.height = canvasSize.height;

        // Fill with white background
        tempCtx.fillStyle = 'white';
        tempCtx.fillRect(0, 0, canvasSize.width, canvasSize.height);

        if (box) {
            const offsetX = (canvasSize.width - box.width) / 2;
            const offsetY = (canvasSize.height - box.height) / 2;

            // Draw only the centered portion onto the temp canvas
            tempCtx.drawImage(
                canvas,
                box.x, box.y, box.width, box.height,
                offsetX, offsetY, box.width, box.height
            );
        } else {
            // If nothing is drawn, use the original canvas
            tempCtx.drawImage(canvas, 0, 0);
        }

        const centeredData = tempCtx.getImageData(0, 0, canvasSize.width, canvasSize.height);
        const data = centeredData.data;

        const result = Array(gridSize).fill(0).map(() => Array(gridSize).fill(0));

        const cellWidth = canvasSize.width / gridSize;
        const cellHeight = canvasSize.height / gridSize;

        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                let sum = 0;
                let count = 0;

                for (let x = Math.floor(j * cellWidth); x < Math.floor((j + 1) * cellWidth); x++) {
                    for (let y = Math.floor(i * cellHeight); y < Math.floor((i + 1) * cellHeight); y++) {
                        const idx = (y * canvasSize.width + x) * 4;
                        const gray = 1 - (data[idx] + data[idx + 1] + data[idx + 2]) / (3 * 255);
                        sum += gray;
                        count++;
                    }
                }

                result[i][j] = count > 0 ? sum / count : 0;
            }
        }

        return result;
    }

    function debounce(func, delay) {
        let timeoutId;
        return function (...args) {
            clearTimeout(timeoutId);
            timeoutId = setTimeout(() => func.apply(this, args), delay);
        };
    }

    async function classifyDoodle() {
        showLoadingState();

        const doodleData = canvasToArray();

        try {
            const response = await fetch('/classify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ drawing: doodleData }),
            });

            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }

            const results = await response.json();
            displayResults(results);
        } catch (error) {
            console.error('Error classifying doodle:', error);
        }
    }

    const debouncedClassify = debounce(classifyDoodle, 500);

    function showLoadingState() {
        let loadingRows = '';
        for (let i = 0; i < 5; i++) {
            loadingRows += `
                <tr>
                    <td><div class="loading-placeholder">Loading...</div></td>
                    <td><div class="loading-placeholder"></div></td>
                </tr>
            `;
        }
        resultsBody.innerHTML = loadingRows;
    }

    function displayResults(results) {
        const topResults = Object.entries(results)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

        if (topResults.length === 0) {
            resultsBody.innerHTML = `
                <tr>
                    <td colspan="2">No results found</td>
                </tr>
            `;
            return;
        }

        let tableRows = '';
        topResults.forEach(([category, probability]) => {
            const probabilityPercent = (probability * 100).toFixed(2);
            tableRows += `
                <tr>
                    <td>${category}</td>
                    <td>
                        ${probabilityPercent}%
                        <div class="probability-bar">
                            <div class="probability-fill" style="width: ${probabilityPercent}%;"></div>
                        </div>
                    </td>
                </tr>
            `;
        });

        resultsBody.innerHTML = tableRows;
    }

    function startDrawing(e) {
        isDrawing = true;
        draw(e);
    }

    function stopDrawing() {
        if (isDrawing) {
            isDrawing = false;
            ctx.beginPath();
            debouncedClassify();
        }
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX || e.touches[0].clientX) - rect.left;
        const y = (e.clientY || e.touches[0].clientY) - rect.top;

        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    clearButton.addEventListener('click', clearCanvas);
});