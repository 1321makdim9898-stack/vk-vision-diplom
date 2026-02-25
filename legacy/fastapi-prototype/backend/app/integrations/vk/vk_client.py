from __future__ import annotations
import os
from typing import Any, Dict, Optional
import requests

VK_API = "https://api.vk.com/method/"
VK_VERSION = "5.199"

class VkClient:
    """
    IMPORTANT:
      - In many cases, user tokens / app type restrictions will block photos.get.
      - We keep this module to demonstrate integration and to support public photos when allowed.
    """
    def __init__(self, access_token: str):
        self.access_token = access_token

    def call(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(params)
        p["access_token"] = self.access_token
        p["v"] = VK_VERSION
        r = requests.get(VK_API + method, params=p, timeout=25)
        data = r.json()
        if "error" in data:
            e = data["error"]
            raise RuntimeError(f"VK API error ({e.get('error_code')}): {e.get('error_msg')}")
        return data["response"]

    def resolve_screen_name(self, screen_name: str) -> Dict[str, Any]:
        return self.call("utils.resolveScreenName", {"screen_name": screen_name})

    def get_profile_photos(self, owner_id: int, count: int = 3) -> Dict[str, Any]:
        # may fail depending on profile type / token
        return self.call("photos.get", {"owner_id": owner_id, "album_id": "profile", "rev": 1, "count": count})
