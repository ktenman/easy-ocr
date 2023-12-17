# easy-ocr
```shell
launchctl unload ~/Library/LaunchAgents/com.user.EasyOcrService.plist
launchctl load ~/Library/LaunchAgents/com.user.EasyOcrService.plist
launchctl start com.user.EasyOcrService     
```
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
    <dict>
        <key>Label</key>
        <string>com.user.EasyOcrService</string>

        <key>RunAtLoad</key>
        <true/>

        <key>KeepAlive</key>
        <true/>

        <key>ProgramArguments</key>
        <array>
            <string>/bin/bash</string>
            <string>/Users/tenman/easy-ocr/start_easy_ocr_service.sh</string>
        </array>

        <key>StandardOutPath</key>
        <string>/Users/tenman/easy-ocr/launchagent-out.log</string>

        <key>StandardErrorPath</key>
        <string>/Users/tenman/easy-ocr/launchagent-err.log</string>

    </dict>
</plist>
```
