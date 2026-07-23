# VK Image Enhancement

Веб-приложение улучшения изображения по следующим параметрам: яркость, контрастность и цветность - посредством запуска в браузере пользователя ML-модели.

## Работа

Поддерживаемые форматы изображений: JPG, PNG, BMP и HEIC/HEIF (для HEIC приложение автоматически преобразует файл в PNG в браузере). Максимальный размер изображения - 15 Мпк.

Веб-приложение доступно по [ссылке](https://busyaska.github.io/VK-Image-Enhancement/).

Пример улучшения изображения:<br>
<img width="2559" height="1351" alt="Image" src="https://github.com/user-attachments/assets/0e5e7f92-12f4-4e77-9c02-c32dc6db80c7" />

## Запуск

Требования:
- Node.js 20 или новее.

### Локальный запуск
```bash
npm install
npm run dev
```
Веб-приложение будет доступно по ссылке: <code>http://localhost:5173</code>.

### Production сборка
```bash
npm run build
```
\- результат будет в `dist`.

