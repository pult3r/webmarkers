<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\ColorDetectController;

Route::get('/color-detect', [ColorDetectController::class, 'showForm'])->name('color.detect.form');
Route::post('/color-detect', [ColorDetectController::class, 'analyze'])->name('color.detect.analyze');


Route::get('/', function () {
    return view('welcome');
});

