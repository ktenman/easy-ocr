# easy-ocr
```shell
launchctl load ~/Library/LaunchAgents/com.user.start_easy_ocr_service.plist
launchctl unload ~/Library/LaunchAgents/com.user.start_easy_ocr_service.plist
```
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.user.start_easy_ocr_service</string>
        <key>RunAtLoad</key>
        <true/>
        <key>Program</key>
        <string>/Users/tenman/easy-ocr/start_easy_ocr_service.sh</string>
    </dict>
</plist>

```
