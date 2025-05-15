<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Http\Client\RequestException;

class ColorDetectController extends Controller
{
    public function showForm()
    {
        return view('color_detect');
    }

    public function analyze(Request $request)
    {
        $request->validate([
            'photo' => 'required|image|max:5120', // max 5MB
        ]);

        $photoPath = $request->file('photo')->getPathname();

        // Wysyłanie zdjęcia do serwisu Pythonowego

        $numColors = 10 ;
        
        try {
            $response = Http::timeout(300)->attach(
                'file', file_get_contents($photoPath), $request->file('photo')->getClientOriginalName()
            )->post("http://webmarkers-yolo-api:8000/color-detect?num_colors={$numColors}");

            if (!$response->successful()) {
                return back()->withErrors(['msg' => 'Błąd połączenia z serwisem analizy.']);
            }

            $colors = $response->json()['dominant_colors_rgb'];

            return view('color_detect', [
                'image' => $request->file('photo')->store('uploads', 'public'),
                'colors' => $colors,
            ]);

        } catch (RequestException $e) {
            return response()->json([
                'error' => 'Serwis analizy kolorów nie odpowiada. Spróbuj ponownie później.',
                'details' => $e->getMessage()
            ], 504); // Gateway Timeout
        }



        
    }
}
