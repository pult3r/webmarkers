<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <title>Detekcja Kolorów</title>
    <style>
        .color-box {
            width: 100px;
            height: 100px;
            display: inline-block;
            margin: 10px;
            border: 1px solid #000;
        }
    </style>
</head>
<body>
    <h1>Wykrywanie kolorów dominujących</h1>

    @if ($errors->any())
        <div style="color: red;">
            @foreach ($errors->all() as $error)
                <p>{{ $error }}</p>
            @endforeach
        </div>
    @endif

    <form method="POST" enctype="multipart/form-data" action="{{ route('color.detect.analyze') }}">
        @csrf
        <input type="file" name="photo" required>
        <button type="submit">Prześlij zdjęcie</button>
    </form>

    @if (!empty($image))
        <h2>Przesłane zdjęcie:</h2>
        <img src="{{ asset('storage/' . $image) }}" alt="Zdjęcie" style="max-width: 400px; display:block; margin: 20px 0;">

        <h2>Dominujące kolory:</h2>
        @foreach ($colors as $color)
            <div class="color-box" style="background-color: {{ $color }};" title="{{ $color }}"></div>
        @endforeach
    @endif
</body>
</html>
