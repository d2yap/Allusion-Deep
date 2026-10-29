import { observer } from 'mobx-react-lite';
import React from 'react';
import { Toggle } from 'widgets';
import { useStore } from '../../contexts/StoreContext';

export const StartupBehavior = observer(() => {
  const { uiStore } = useStore();

  //const [isAutoUpdateEnabled, setAutoUpdateEnabled] = useState(
  //  RendererMessenger.isCheckUpdatesOnStartupEnabled,
  //);

  return (
    <div className="vstack">
      <Toggle
        checked={uiStore.isRememberSearchEnabled}
        onChange={uiStore.toggleRememberSearchQuery}
      >
        Restore and query last submitted search query
      </Toggle>
    </div>
  );
});
