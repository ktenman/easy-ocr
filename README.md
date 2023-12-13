# easy-ocr
```shell
launchctl load ~/Library/LaunchAgents/com.user.easyocr.plist
launchctl unload ~/Library/LaunchAgents/com.user.easyocr.plist
```
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.user.easyocr</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/tenman/easy-ocr/start.sh</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```
