import os
import pathlib
import requests
from typing import List, Dict, Any, Optional

# Базовый URL VK API
VK_API_URL = "https://api.vk.com/method"
VK_API_VERSION = "5.199"

# Папка, куда будут сохраняться фото
OUTPUT_DIR = pathlib.Path("vk_import")
OUTPUT_DIR.mkdir(exist_ok=True)


class VkApiError(RuntimeError):
    """Простое исключение для ошибок VK API."""
    pass


def vk_api_call(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вызов метода VK API с простой обработкой ошибок.
    """
    url = f"{VK_API_URL}/{method}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        err = data["error"]
        raise VkApiError(
            f"VK API error {err.get('error_code')}: {err.get('error_msg')}"
        )

    return data["response"]


def resolve_owner_id(user_or_screen_name: str, access_token: str) -> int:
    """
    Преобразуем @screen_name или числовой id в числовой owner_id.
    """
    # Если похоже на число – просто возвращаем
    if user_or_screen_name.isdigit():
        return int(user_or_screen_name)

    params = {
        "screen_name": user_or_screen_name.lstrip("@"),
        "access_token": access_token,
        "v": VK_API_VERSION,
    }
    resp = vk_api_call("utils.resolveScreenName", params)
    if not resp or "object_id" not in resp:
        raise VkApiError(
            f"Не удалось преобразовать '{user_or_screen_name}' в owner_id."
        )
    return int(resp["object_id"])


def get_profile_photos(owner_id: int, access_token: str, count: int = 5) -> List[Dict[str, Any]]:
    """
    Пытаемся получить фотографии профиля.
    Сначала photos.get c album_id='profile',
    при ошибке 1051/несовместимости можно добавить сюда другие варианты.
    """
    params = {
        "owner_id": owner_id,
        "album_id": "profile",
        "rev": 1,              # от новых к старым
        "extended": 0,
        "count": count,
        "access_token": access_token,
        "v": VK_API_VERSION,
    }

    try:
        resp = vk_api_call("photos.get", params)
        items = resp.get("items", [])
        if not items:
            print("[VK] Не найдено фото в альбоме 'profile'.")
        return items

    except VkApiError as e:
        # Если нужно – можно сюда добавить fallback на photos.getUserPhotos и т.п.
        print(f"[VK] Ошибка при вызове photos.get: {e}")
        raise


def download_photo(photo_item: Dict[str, Any], index: int) -> Optional[pathlib.Path]:
    """
    Скачиваем одно фото (берём максимальный по размеру url) в OUTPUT_DIR.
    """
    sizes = photo_item.get("sizes") or []
    if not sizes:
        return None

    # Берём максимальный по ширине размер
    best = max(sizes, key=lambda s: s.get("width", 0))
    url = best.get("url")
    if not url:
        return None

    ext = ".jpg"
    filename = OUTPUT_DIR / f"vk_photo_{index:03d}{ext}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(filename, "wb") as f:
            f.write(r.content)
        return filename
    except Exception as e:
        print(f"[VK] Не удалось скачать фото #{index}: {e}")
        return None


def fetch_vk_photos(user_or_screen_name: str, access_token: str, count: int = 3) -> List[pathlib.Path]:
    """
    Основная функция: по id/@screen_name и токену скачивает count фото профиля.
    Возвращает список путей к сохранённым файлам.
    """
    print("[VK] Определяем owner_id…")
    owner_id = resolve_owner_id(user_or_screen_name, access_token)
    print(f"[VK] owner_id = {owner_id}")

    print(f"[VK] Получаем до {count} фото профиля…")
    items = get_profile_photos(owner_id, access_token, count=count)

    saved_paths: List[pathlib.Path] = []
    for idx, item in enumerate(items, start=1):
        p = download_photo(item, idx)
        if p:
            saved_paths.append(p)
            print(f"[VK] Сохранено: {p}")

    print(f"[VK] Всего сохранено файлов: {len(saved_paths)}")
    return saved_paths


if __name__ == "__main__":
    print("=== VK Fetch Photos (profile) ===")
    user = input("Введите VK id или @screen_name: ").strip()
    token = input("Введите access_token VK: ").strip()
    if not user or not token:
        print("Нужны и id/@screen_name, и access_token. Выход.")
        raise SystemExit(1)

    try:
        count_str = input("Сколько фото профиля получить? [по умолчанию 3]: ").strip()
        count = int(count_str) if count_str else 3
    except ValueError:
        count = 3

    try:
        paths = fetch_vk_photos(user, token, count=count)
        if paths:
            print("\nГотово. Фото лежат в папке:", OUTPUT_DIR.resolve())
        else:
            print("\nФото не были сохранены.")
    except VkApiError as e:
        print(f"\nVK API ошибка: {e}")
    except Exception as e:
        print(f"\nНеожиданная ошибка: {e}")
